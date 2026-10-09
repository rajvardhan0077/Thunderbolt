import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String _defaultBaseUrl = 'http://10.0.2.2:5000';

  final String baseUrl;

  ApiService({String? baseUrl}) : baseUrl = baseUrl ?? _defaultBaseUrl;

  Future<Map<String, dynamic>> getHealth() async {
    final response = await http.get(Uri.parse('$baseUrl/health')).timeout(const Duration(seconds: 10));
    return _decode(response, label: 'health');
  }

  Future<Map<String, dynamic>> predictAvailability(Map<String, dynamic> payload) async {
    return _post('/predict-availability', payload);
  }

  Future<Map<String, dynamic>> predictWait(Map<String, dynamic> payload) async {
    return _post('/predict-wait', payload);
  }

  Future<Map<String, dynamic>> predictPrice(Map<String, dynamic> payload) async {
    return _post('/predict-price', payload);
  }

  Future<Map<String, dynamic>> predictPartDetails(Map<String, dynamic> payload) async {
    return _post('/predict-part-details', payload);
  }

  Future<Map<String, dynamic>> findGarages(Map<String, dynamic> payload) async {
    return _post('/find-garages', payload);
  }

  Future<Map<String, dynamic>> submitServiceRequest(Map<String, dynamic> payload) async {
    return _post('/service-request', payload);
  }

  Future<Map<String, dynamic>> _post(String endpoint, Map<String, dynamic> payload) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl$endpoint'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        )
        .timeout(const Duration(seconds: 15));
    return _decode(response, label: endpoint);
  }

  Map<String, dynamic> _decode(http.Response response, {required String label}) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (response.body.isEmpty) {
        return {'status': 'ok', 'label': label};
      }
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    final decoded = jsonDecode(response.body);
    final errorMessage = decoded is Map<String, dynamic> ? decoded['error'] ?? 'Request failed' : 'Request failed';
    throw Exception(errorMessage);
  }
}
