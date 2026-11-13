"use strict";
const token = location.hash.slice(1) || sessionStorage.getItem("mokuml-capability") || "";
if (token) sessionStorage.setItem("mokuml-capability", token);
history.replaceState(null, "", "/");
const $ = id => document.getElementById(id);
const node = (tag, text, cls) => { const el=document.createElement(tag); if(text!==undefined)el.textContent=String(text);if(cls)el.className=cls;return el; };
let selectedRun=null, shownResult=null, planRevision="";
function notice(message){$("notice").textContent=message;$("notice").hidden=!message;}
async function api(path, options={}) {
 const headers={Authorization:"Bearer "+token,...options.headers};
 if(options.body && !(options.body instanceof FormData))headers["Content-Type"]="application/json";
 const response=await fetch("/api/"+path,{...options,headers}), result=await response.json();
 if(!response.ok || result.ok===false)throw Error(result.message || "This operation could not complete.");
 return result;
}
function table(container, columns, rows){
 const t=node("table"),head=node("tr");columns.forEach(c=>head.append(node("th",c)));
 const thead=node("thead");thead.append(head);t.append(thead);const body=node("tbody");
 rows.forEach(values=>{const row=node("tr");values.forEach(v=>row.append(node("td",v==null?"—":typeof v==="number"?new Intl.NumberFormat(undefined,{maximumFractionDigits:2}).format(v):v)));body.append(row);});
 t.append(body);container.replaceChildren(t);
}
async function save(path, name){
 const response=await fetch("/api/"+path,{headers:{Authorization:"Bearer "+token}});
 if(!response.ok)throw Error("Export unavailable.");
 const url=URL.createObjectURL(await response.blob()), link=node("a");link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function showResult(runId){
 if(shownResult===runId)return;
 const result=await api("runs/"+runId), a=result.augmented;shownResult=runId;selectedRun=runId;
 $("results").hidden=false;$("result-title").textContent=a.target+" · "+a.model.engine;
 $("result-description").textContent=a.result_kind+" · "+a.split+" evaluation · "+a.training_rows+" training / "+a.calibration_rows+" calibration / "+a.test_rows+" test records"+(a.prediction_window_start?" · "+a.prediction_window_start+" to "+a.prediction_window_end:"");
 $("metrics").replaceChildren();Object.entries(a.metrics).forEach(([key,value])=>{const block=node("div",undefined,"metric");block.append(node("strong",Number(value).toPrecision(4)),node("span",key));$("metrics").append(block);});
 $("warnings").replaceChildren(...a.data_quality.concat(a.limitations).map(text=>node("p",text,"muted")));
 const rows=result.predictions.slice(0,100);
 table($("predictions"),["Case","Identifiers","Prediction"+(a.units?" ("+a.units+")":""),"Lower","Upper","Class probabilities"],rows.map(r=>[r.case_id,JSON.stringify(r.identifiers),r.prediction,r.lower,r.upper,r.probabilities?Object.entries(r.probabilities).map(([k,v])=>k+": "+(v*100).toFixed(1)+"%").join(", "):"—"]));
}
function buildPlans(plans){
 const revision=JSON.stringify(plans.map(p=>p.id));if(revision===planRevision)return;planRevision=revision;$("plans").replaceChildren();
 plans.forEach(plan=>{
  const box=node("details"),summary=node("summary",plan.target+" · "+plan.split);box.append(summary,node("pre",plan.sql));
  if(!plan.horizon){const form=node("form");form.append(node("p","Enter one private case using the prepared feature definitions. Leave an input blank for a missing value."));
   plan.feature_names.forEach(name=>{const label=node("label",name),input=node("input");input.name=name;input.autocomplete="off";label.append(input);form.append(label);});
   const submit=node("button","Save private case");submit.type="submit";form.append(submit);
   form.onsubmit=async e=>{e.preventDefault();const values={};new FormData(form).forEach((v,k)=>{if(v==="")values[k]=null;else{try{values[k]=JSON.parse(v);}catch{values[k]=v;}}});try{await api("cases/"+plan.id,{method:"POST",body:JSON.stringify(values)});notice("Case saved. Ask ChatGPT to run the prepared plan.");form.reset();}catch(err){notice(err.message);}};
   box.append(form);
  }$("plans").append(box);
 });
}
async function refresh(){
 if(!token){notice("Open this private window from the MokuML plugin.");return;}
 try{
  const state=await api("state"),catalog=state.catalog;
  $("source-state").textContent=catalog?catalog.table_count+" relations available to ChatGPT":"No source connected.";
  $("relations").replaceChildren();if(catalog)catalog.tables.forEach(t=>{const button=node("button",t.name+" · "+t.columns.length+" columns","quiet");button.onclick=async()=>{try{await api("database/load",{method:"POST",body:JSON.stringify({table_id:t.id})});await refresh();}catch(e){notice(e.message);}};$("relations").append(button);});
  const m=state.models;$("model-state").textContent=m.downloading?"Downloading and verifying models…":["TabICL: "+(m.tabicl?"ready":"missing"),"Kumo: "+(m.kumo?"ready":"missing"),"TabPFN: "+(m.tabpfn?"ready":"missing"),"Qwen: "+(m.qwen?"ready":"missing")].join(" · ");
  if(state.setup_error)notice("Model setup failed. Check connectivity, access, license and free disk space.");
  buildPlans(state.plans);
  $("preview-section").hidden=!state.dataset;
  if(state.dataset){const columns=state.dataset.columns.slice(0,12).map(c=>c.name);table($("preview"),columns,state.preview.map(r=>columns.map(c=>r[c])));}
  $("no-runs").hidden=state.runs.length>0;$("runs").replaceChildren();
  for(const reply of state.runs){const run=reply.run,row=node("div",undefined,"run-row");row.append(node("span",run.task+" · "+run.status+" · "+run.stage));
   if(run.status==="running"){const cancel=node("button","Cancel","quiet");cancel.onclick=async()=>{await api("runs/"+run.id+"/cancel",{method:"POST"});await refresh();};row.append(cancel);}
   if(run.status==="completed"){const view=node("button","View results","secondary");view.onclick=()=>showResult(run.id).catch(e=>notice(e.message));row.append(view);if(!selectedRun)await showResult(run.id);}
   if(!reply.ok)row.append(node("p",reply.message));$("runs").append(row);
  }
 }catch(e){notice(e.message);}
}
$("file-form").onsubmit=async e=>{e.preventDefault();try{const body=new FormData();body.append("file",$("file").files[0]);body.append("scoring",String($("scoring-file").checked));await api("import",{method:"POST",body});notice("Data added locally. Describe your prediction in ChatGPT.");await refresh();}catch(err){notice(err.message);}};
$("database-kind").onchange=()=>{$("server-fields").hidden=$("database-kind").value==="sqlite";document.querySelector('[name="port"]').value=$("database-kind").value==="mysql"?"3306":"5432";};
$("database-form").onsubmit=async e=>{e.preventDefault();try{const data=Object.fromEntries(new FormData(e.target));if(data.kind==="sqlite")await api("sqlite/open",{method:"POST"});else{data.port=Number(data.port);data.schema_name=data.schema_name||null;data.require_tls=new FormData(e.target).has("require_tls");await api("database/connect",{method:"POST",body:JSON.stringify(data)});}e.target.querySelector('[name="password"]').value="";notice("Connected. Ask ChatGPT to inspect the source.");await refresh();}catch(err){notice(err.message);}};
const setup=async body=>{try{await api("models",{method:"POST",body:JSON.stringify(body)});await refresh();}catch(e){notice(e.message);}};
$("setup-base").onclick=()=>setup({component:"all"});$("setup-text").onclick=()=>setup({component:"qwen"});
$("restore-v2").onclick=()=>setup({component:"tabpfn"});
$("license-form").onsubmit=async e=>{e.preventDefault();const secret=$("model-token").value;$("model-token").value="";await setup({component:"tabpfn35",licensed_opt_in:$("license").checked,token:secret});};
$("release").onclick=async()=>{try{await api("release",{method:"POST"});selectedRun=shownResult=null;$("results").hidden=true;notice("Data, plans and results released.");await refresh();}catch(e){notice(e.message);}};
$("download-csv").onclick=()=>save("runs/"+selectedRun+"/csv","mokuml-predictions.csv").catch(e=>notice(e.message));
$("download-report").onclick=()=>save("runs/"+selectedRun+"/report","mokuml-prediction-report.json").catch(e=>notice(e.message));
refresh();setInterval(refresh,2500);
