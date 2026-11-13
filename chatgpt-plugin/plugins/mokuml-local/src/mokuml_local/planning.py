"""Compile typed relational recipes; no model-supplied SQL is executed."""

from __future__ import annotations

import uuid
from datetime import timedelta, timezone

import sqlglot
from dateutil.relativedelta import relativedelta
from sqlglot import exp

from .contracts import PredictionTask, PreparedPredictionPlan
from .data import MAX_ROWS
from .errors import SafeError


def references(expression):
    result = {expression.column_id} if expression.kind == "column" else set()
    for arg in expression.args:
        result |= references(arg)
    return result


def has_aggregate(expression):
    return expression.kind == "aggregate" or any(
        has_aggregate(a) for a in expression.args
    )


def identity_references(expression):
    """Counts of identifiers are derived quantities, not identifier inputs."""
    if expression.kind == "aggregate" and expression.function in {
        "count",
        "count_distinct",
    }:
        return set()
    result = {expression.column_id} if expression.kind == "column" else set()
    for arg in expression.args:
        result |= identity_references(arg)
    return result


def literal(value):
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    return exp.Literal.string(value)


def expression_ast(expression, columns, depth=0):
    if depth > 16:
        raise SafeError("invalid_plan")
    if expression.kind == "column":
        if expression.column_id not in columns:
            raise SafeError("invalid_plan")
        return columns[expression.column_id].copy()
    if expression.kind == "literal":
        return literal(expression.value)
    args = [expression_ast(a, columns, depth + 1) for a in expression.args]
    if expression.kind == "aggregate":
        if any(has_aggregate(a) for a in expression.args):
            raise SafeError("invalid_plan")
        if expression.function == "count":
            return exp.Count(this=args[0] if args else exp.Star())
        if not args:
            raise SafeError("invalid_plan")
        if expression.function == "count_distinct":
            return exp.Count(this=exp.Distinct(expressions=args))
        return {"sum": exp.Sum, "mean": exp.Avg, "min": exp.Min, "max": exp.Max}[
            expression.function
        ](this=args[0])
    unary = {
        "is_null": lambda a: exp.Is(this=a, expression=exp.Null()),
        "is_not_null": lambda a: exp.Not(this=exp.Is(this=a, expression=exp.Null())),
        "abs": lambda a: exp.Abs(this=a),
    }
    binary = {
        "add": exp.Add,
        "subtract": exp.Sub,
        "multiply": exp.Mul,
        "equal": exp.EQ,
        "not_equal": exp.NEQ,
        "less": exp.LT,
        "less_equal": exp.LTE,
        "greater": exp.GT,
        "greater_equal": exp.GTE,
        "and": exp.And,
        "or": exp.Or,
    }
    if expression.operator in unary and len(args) == 1:
        return unary[expression.operator](args[0])
    if expression.operator in binary and len(args) == 2:
        return exp.Paren(
            this=binary[expression.operator](this=args[0], expression=args[1])
        )
    if expression.operator == "divide" and len(args) == 2:
        return exp.Div(
            this=args[0], expression=exp.Nullif(this=args[1], expression=literal(0))
        )
    if expression.operator == "coalesce" and len(args) >= 2:
        return exp.Coalesce(this=args[0], expressions=args[1:])
    if expression.operator == "case" and len(args) == 3:
        return exp.Case(ifs=[exp.If(this=args[0], true=args[1])], default=args[2])
    raise SafeError("invalid_plan")


def validate_sql(sql, dialect, allowed_locations):
    """Fail closed: AST parser is required, including for compiled queries."""
    try:
        roots = sqlglot.parse(sql, read=dialect)
        if len(roots) != 1 or not isinstance(roots[0], (exp.Select, exp.Union)):
            raise SafeError("unsafe_query")
        root = roots[0]
        ctes = {c.alias_or_name for c in root.find_all(exp.CTE)}
        forbidden = (
            "Insert",
            "Update",
            "Delete",
            "Create",
            "Drop",
            "Alter",
            "Command",
            "Into",
            "Copy",
            "Lock",
            "Transaction",
            "Use",
            "Pragma",
            "Attach",
            "Merge",
        )
        for node in root.walk():
            if type(node).__name__ in forbidden:
                raise SafeError("unsafe_query")
            if isinstance(node, exp.Table):
                if not isinstance(node.this, exp.Identifier):
                    raise SafeError("unsafe_query")
                if node.catalog:
                    raise SafeError("unsafe_query")
                if (node.db or None, node.name) not in allowed_locations and not (
                    node.name in ctes and not node.db
                ):
                    raise SafeError("unsafe_query")
            if isinstance(node, exp.Anonymous) and node.name.lower() != "datetime":
                raise SafeError("unsafe_query")
            # Every function must be one emitted by this compiler.
            if isinstance(node, exp.Func) and not isinstance(
                node,
                (
                    exp.Sum,
                    exp.Avg,
                    exp.Min,
                    exp.Max,
                    exp.Count,
                    exp.Coalesce,
                    exp.Nullif,
                    exp.Abs,
                    exp.Cast,
                    exp.Case,
                    exp.If,
                    exp.Anonymous,
                    exp.And,
                    exp.Or,
                ),
            ):
                raise SafeError("unsafe_query")
        return root
    except SafeError:
        raise
    except Exception:
        raise SafeError("unsafe_query") from None


def shift(value, n, unit):
    if unit == "month":
        return value + relativedelta(months=n)
    return value + timedelta(days=n * (7 if unit == "week" else 1))


def next_boundary(value, unit):
    if unit == "month":
        return value.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) + relativedelta(months=1)
    floor = value.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "week":
        return floor + timedelta(days=7 - floor.weekday())
    return floor + timedelta(days=1)


class Compiler:
    def __init__(self, catalog):
        self.catalog = catalog
        self.source = catalog.source
        self.dialect = self.source.dialect

    def relation(self, table_id, alias):
        if table_id not in self.catalog.tables:
            raise SafeError("invalid_plan")
        schema, name = self.source.table_location(table_id)
        return exp.Table(
            this=exp.to_identifier(name, quoted=True),
            db=exp.to_identifier(schema, quoted=True) if schema else None,
        ).as_(alias)

    def base(self, query, extra_table=None):
        table_ids = [query.table_id] + [j.table_id for j in query.joins]
        if len(set(table_ids)) != len(table_ids):
            raise SafeError("invalid_plan")
        columns = {}
        for i, key in enumerate(table_ids):
            if key not in self.catalog.tables:
                raise SafeError("invalid_plan")
            columns.update(
                {
                    c.id: exp.column(c.name, table=f"t{i}", quoted=True)
                    for c in self.catalog.tables[key].columns
                }
            )
        result = exp.select().from_(self.relation(table_ids[0], "t0"))
        available = {c.id for c in self.catalog.tables[table_ids[0]].columns}
        for i, join in enumerate(query.joins, 1):
            new = {c.id for c in self.catalog.tables[join.table_id].columns}
            predicates = []
            for left, right in join.column_pairs:
                if not (
                    (left in available and right in new)
                    or (right in available and left in new)
                ):
                    raise SafeError("invalid_plan")
                predicates.append(
                    exp.EQ(this=columns[left].copy(), expression=columns[right].copy())
                )
            result = result.join(
                self.relation(join.table_id, f"t{i}"),
                on=exp.and_(*predicates),
                join_type=join.kind,
            )
            available |= new
        if query.filters:
            if any(has_aggregate(f) for f in query.filters):
                raise SafeError("invalid_plan")
            result = result.where(
                exp.and_(*(expression_ast(f, columns) for f in query.filters))
            )
        return result, columns

    def timestamp(self, node):
        if self.dialect == "sqlite":
            return exp.Anonymous(this="datetime", expressions=[node])
        if self.dialect in {"duckdb", "postgres"}:
            return exp.Cast(
                this=exp.Cast(this=node, to=exp.DataType.build("TIMESTAMPTZ")),
                to=exp.DataType.build("TIMESTAMP"),
            )
        return exp.Cast(
            this=node,
            to=exp.DataType.build(
                "DATETIME" if self.dialect == "mysql" else "TIMESTAMP"
            ),
        )

    def future_query(self, task):
        future = task.future
        physical_names = {
            self.source.table_location(key)[1].casefold() for key in self.catalog.tables
        }

        def cte_name(stem):
            name = stem
            while name.casefold() in physical_names:
                name += "_"
            physical_names.add(name.casefold())
            return name

        events_name = cte_name("__moku_events")
        anchors_name = cte_name("__moku_anchors")
        base, columns = self.base(task.query)
        if task.query.group_by_column_ids or not has_aggregate(task.target.expression):
            raise SafeError("invalid_plan")
        if any(not has_aggregate(f.expression) for f in task.features):
            raise SafeError("invalid_plan")
        used = references(task.target.expression) | {future.event_time_column_id}
        for feature in task.features:
            used |= references(feature.expression)
        if future.entity_column_id:
            used.add(future.entity_column_id)
        if not used <= columns.keys():
            raise SafeError("invalid_plan")
        base = base.select(
            *(columns[c].copy().as_("c_" + c, quoted=True) for c in sorted(used))
        )

        # Normalize all anchors to UTC; no session timezone changes are required.
        def utc(value):
            return (
                value.astimezone(timezone.utc).replace(tzinfo=None)
                if value.tzinfo
                else value
            )

        observed, start = utc(future.observed_through), utc(future.start)
        anchors, position = [], 0
        while True:
            cutoff = shift(observed, -position, future.unit)
            if cutoff < start:
                break
            label_start = (
                next_boundary(cutoff, future.unit)
                if future.alignment == "calendar_next"
                else cutoff
            )
            label_end = shift(label_start, future.horizon, future.unit)
            fields = {
                "__as_of": cutoff,
                "__label_start": label_start,
                "__label_end": label_end,
            }
            fields.update(
                {
                    f"__past_{n}": shift(cutoff, -n, future.unit)
                    for n in future.lookback_periods
                }
            )
            anchors.append(
                exp.select(
                    *(
                        self.timestamp(literal(v.isoformat(sep=" "))).as_(
                            k, quoted=True
                        )
                        for k, v in fields.items()
                    )
                )
            )
            position += 1
            if position > 1200:
                raise SafeError("resource_limit")
        anchor_query = anchors[0]
        for anchor in anchors[1:]:
            anchor_query = exp.union(anchor_query, anchor, distinct=False)

        def a(name):
            return exp.column(name, table="a", quoted=True)

        event_columns = {c: exp.column("c_" + c, table="e", quoted=True) for c in used}
        event_time = self.timestamp(event_columns[future.event_time_column_id].copy())
        entity = future.entity_column_id

        def aggregate(recipe, lower, upper):
            conditions = [
                exp.GTE(this=event_time.copy(), expression=lower),
                exp.LT(this=event_time.copy(), expression=upper),
            ]
            if entity:
                conditions.append(
                    exp.EQ(
                        this=event_columns[entity].copy(),
                        expression=exp.column("__entity", table="g", quoted=True),
                    )
                )
            query = (
                exp.select(expression_ast(recipe, event_columns))
                .from_(f"{events_name} AS e")
                .where(exp.and_(*conditions))
            )
            # Empty complete windows are zero only for additive/count outcomes.
            subquery = exp.Subquery(this=query)
            if recipe.kind == "aggregate" and recipe.function in {
                "sum",
                "count",
                "count_distinct",
            }:
                return exp.Coalesce(this=subquery, expressions=[literal(0)])
            return subquery

        label = aggregate(task.target.expression, a("__label_start"), a("__label_end"))
        target = exp.Case(
            ifs=[
                exp.If(
                    this=exp.LTE(
                        this=a("__label_end"),
                        expression=self.timestamp(literal(observed.isoformat(sep=" "))),
                    ),
                    true=label,
                )
            ],
            default=exp.Null(),
        )
        outputs = [target.as_(task.target.name, quoted=True)]
        feature_names = []
        for feature in task.features:
            for n in future.lookback_periods:
                name = f"{feature.name}_past_{n}_{future.unit}"
                feature_names.append(name)
                outputs.append(
                    aggregate(feature.expression, a(f"__past_{n}"), a("__as_of")).as_(
                        name, quoted=True
                    )
                )
        # Calendar covariates are deterministic and do not consume future records.
        feature_names += ["calendar_month", "calendar_weekday"]
        # Values are attached to anchors in Python for portable SQL.
        # SQL temporal functions vary significantly between the supported databases.
        for name in ["__as_of", "__label_start", "__label_end"]:
            outputs.append(a(name).as_(name, quoted=True))
        if entity:
            outputs.append(
                exp.column("__entity", table="g", quoted=True).as_(
                    "__entity", quoted=True
                )
            )
        query = exp.select(*outputs).from_(f"{anchors_name} AS a")
        if entity:
            entities = (
                exp.select(
                    exp.column("c_" + entity, quoted=True).as_("__entity", quoted=True)
                )
                .distinct()
                .from_(events_name)
            )
            query = query.join(entities.subquery("g"), join_type="CROSS")
            # An entity must have existed at each historical cutoff.
            exists = (
                exp.select(literal(1))
                .from_(f"{events_name} AS e")
                .where(
                    exp.and_(
                        exp.EQ(
                            this=event_columns[entity].copy(),
                            expression=exp.column("__entity", table="g", quoted=True),
                        ),
                        exp.LT(this=event_time.copy(), expression=a("__as_of")),
                    )
                )
            )
            query = query.where(exp.Exists(this=exists))
        query = query.with_(events_name, as_=base).with_(anchors_name, as_=anchor_query)
        return query, feature_names, ["__entity"] if entity else []

    def compile(self, raw_task):
        try:
            task = PredictionTask.model_validate(raw_task)
            if task.group_column_id and task.time_column_id:
                raise SafeError("invalid_split")
            source_columns = self.catalog.columns
            target_refs = references(task.target.expression)
            if not target_refs and not has_aggregate(task.target.expression):
                raise SafeError("invalid_plan")
            feature_refs = set().union(
                *(references(f.expression) for f in task.features)
            )
            identities = set(task.identifier_column_ids) | {
                c.id
                for table in self.catalog.tables.values()
                for c in table.columns
                if c.primary_key
            }
            if task.group_column_id:
                identities.add(task.group_column_id)
            if task.future and task.future.entity_column_id:
                identities.add(task.future.entity_column_id)
            blocked = set(task.unavailable_column_ids)
            if not task.future:
                blocked |= target_refs
                if identity_references(task.target.expression) & identities:
                    raise SafeError("invalid_plan")
                if any(f.expression == task.target.expression for f in task.features):
                    raise SafeError("invalid_plan")
            if feature_refs & blocked:
                raise SafeError("invalid_plan")
            if any(
                identity_references(f.expression) & identities for f in task.features
            ):
                raise SafeError("invalid_plan")
            if any(
                references(f) & set(task.unavailable_column_ids)
                for f in task.query.filters
            ):
                raise SafeError("invalid_plan")
            names = [task.target.name] + [f.name for f in task.features]
            if len(set(n.casefold() for n in names)) != len(names) or any(
                n.startswith("__") for n in names
            ):
                raise SafeError("invalid_plan")
            all_ids = (
                target_refs
                | feature_refs
                | set(task.identifier_column_ids)
                | set(task.unavailable_column_ids)
            )
            all_ids |= {c for c in [task.time_column_id, task.group_column_id] if c}
            if not all_ids <= source_columns.keys():
                raise SafeError("invalid_plan")
            if (
                task.query.joins
                and not task.future
                and not task.identifier_column_ids
                and not task.query.group_by_column_ids
            ):
                raise SafeError("invalid_plan")
            scoring_sql = None
            if task.future:
                if task.scoring_table_id or task.time_column_id or task.group_column_id:
                    raise SafeError("invalid_plan")
                query, feature_names, identifiers = self.future_query(task)
                split, horizon = (
                    "temporal",
                    f"{task.future.horizon} {task.future.unit} ({task.future.alignment})",
                )
            else:
                query, columns = self.base(task.query)
                outputs = [
                    expression_ast(task.target.expression, columns).as_(
                        task.target.name, quoted=True
                    )
                ]
                outputs += [
                    expression_ast(f.expression, columns).as_(f.name, quoted=True)
                    for f in task.features
                ]
                identifiers = [
                    f"__id_{i}" for i, _ in enumerate(task.identifier_column_ids)
                ]
                roles = list(zip(task.identifier_column_ids, identifiers))
                if task.group_column_id:
                    roles.append((task.group_column_id, "__group"))
                if task.time_column_id:
                    roles.append((task.time_column_id, "__as_of"))
                outputs += [
                    columns[c].copy().as_(alias, quoted=True) for c, alias in roles
                ]
                query = query.select(*outputs)
                if task.query.group_by_column_ids:
                    query = query.group_by(
                        *(columns[c].copy() for c in task.query.group_by_column_ids)
                    )
                feature_names = [f.name for f in task.features]
                split = (
                    "temporal"
                    if task.time_column_id
                    else "grouped"
                    if task.group_column_id
                    else "stratified"
                    if task.task == "classification"
                    else "random"
                )
                horizon = None
                if task.scoring_table_id:
                    scoring_sql = self.scoring_query(task)
            output_names = [task.target.name] + feature_names + identifiers
            if len({n.casefold() for n in output_names}) != len(output_names):
                raise SafeError("invalid_plan")
            query = query.limit(MAX_ROWS + 1)
            sql = query.sql(dialect=self.dialect, pretty=True)
            allowed = {self.source.table_location(key) for key in self.catalog.tables}
            validate_sql(sql, self.dialect, allowed)
            if scoring_sql:
                validate_sql(scoring_sql, self.dialect, allowed)
            plan = PreparedPredictionPlan(
                id=uuid.uuid4().hex,
                source_id=self.catalog.id,
                source_revision=self.catalog.revision,
                task=task.task,
                sql=sql,
                scoring_sql=scoring_sql,
                target=task.target.name,
                feature_names=feature_names,
                identifier_names=identifiers,
                split=split,
                units=task.units,
                horizon=horizon,
                findings=[
                    "Read-only AST validated.",
                    "Input records remain local.",
                    "Feature availability follows the task definition; confirm ambiguous source semantics.",
                    "Evaluation partitions are independent of model selection.",
                ],
            )
            return task, plan
        except SafeError:
            raise
        except Exception:
            raise SafeError("invalid_plan") from None

    def scoring_query(self, task):
        table = self.catalog.tables.get(task.scoring_table_id)
        if table is None:
            raise SafeError("scoring_mismatch")
        if task.query.joins or task.query.group_by_column_ids:
            outputs = [exp.Null().as_(task.target.name, quoted=True)]
            for feature in task.features:
                dest = task.scoring_column_map.get(feature.name)
                if dest is None:
                    matches = [c.id for c in table.columns if c.name == feature.name]
                    dest = matches[0] if len(matches) == 1 else None
                if dest not in {c.id for c in table.columns}:
                    raise SafeError("scoring_mismatch")
                outputs.append(
                    exp.column(
                        self.catalog.columns[dest][1].name, table="s", quoted=True
                    ).as_(feature.name, quoted=True)
                )
            for i, source_id in enumerate(task.identifier_column_ids):
                name = self.catalog.columns[source_id][1].name
                dest = task.scoring_column_map.get(source_id)
                matches = [
                    c
                    for c in table.columns
                    if c.id == dest or (dest is None and c.name == name)
                ]
                node = (
                    exp.column(matches[0].name, table="s", quoted=True)
                    if len(matches) == 1
                    else exp.Null()
                )
                outputs.append(node.as_(f"__id_{i}", quoted=True))
            return (
                exp.select(*outputs)
                .from_(self.relation(table.id, "s"))
                .limit(MAX_ROWS + 1)
                .sql(dialect=self.dialect, pretty=True)
            )
        mapping = {}
        needed = set().union(*(references(f.expression) for f in task.features))
        needed |= set(task.identifier_column_ids)
        for source_id in needed:
            dest = task.scoring_column_map.get(source_id)
            if dest is None:
                name = self.catalog.columns[source_id][1].name
                matches = [c.id for c in table.columns if c.name == name]
                dest = matches[0] if len(matches) == 1 else None
            if dest not in {c.id for c in table.columns}:
                raise SafeError("scoring_mismatch")
            mapping[source_id] = exp.column(
                self.catalog.columns[dest][1].name, table="s", quoted=True
            )
        outputs = [exp.Null().as_(task.target.name, quoted=True)]
        outputs += [
            expression_ast(f.expression, mapping).as_(f.name, quoted=True)
            for f in task.features
        ]
        outputs += [
            mapping[c].copy().as_(f"__id_{i}", quoted=True)
            for i, c in enumerate(task.identifier_column_ids)
        ]
        return (
            exp.select(*outputs)
            .from_(self.relation(table.id, "s"))
            .limit(MAX_ROWS + 1)
            .sql(dialect=self.dialect, pretty=True)
        )
