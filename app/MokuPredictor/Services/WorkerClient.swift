import Foundation

struct WorkerClient {
    var baseURL = WorkerEndpoint.baseURL

    func get<Response: Decodable>(_ path: String) async throws -> Response {
        let request = URLRequest(url: baseURL.appendingPathComponent(path))
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.validate(response: response, data: data)
        return try Self.decoder.decode(Response.self, from: data)
    }

    func post<Response: Decodable>(_ path: String, payload: [String: JSONValue]) async throws -> Response {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try Self.encoder.encode(payload)
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.validate(response: response, data: data)
        return try Self.decoder.decode(Response.self, from: data)
    }

    private static let encoder = JSONEncoder()
    private static let decoder = JSONDecoder()

    private static func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw WorkerClientError.http(status: http.statusCode, body: body)
        }
    }
}

enum WorkerClientError: LocalizedError {
    case http(status: Int, body: String)

    var errorDescription: String? {
        switch self {
        case .http(let status, let body):
            return "Worker returned HTTP \(status). \(body)"
        }
    }
}
