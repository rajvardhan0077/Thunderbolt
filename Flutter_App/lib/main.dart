import 'package:flutter/material.dart';
import 'package:smart_garage_app/services/api_service.dart';

void main() {
  runApp(const SmartGarageApp());
}

class SmartGarageApp extends StatelessWidget {
  const SmartGarageApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart Garage',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepOrange),
        useMaterial3: true,
      ),
      home: const SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(seconds: 2), () {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const LoginScreen()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.deepOrange.shade50,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.build_circle, size: 90, color: Colors.deepOrange),
            const SizedBox(height: 16),
            const Text(
              'Smart Garage',
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              'Thunderbird 350X Spare Parts Assistant',
              style: TextStyle(color: Colors.grey.shade700),
            ),
            const SizedBox(height: 28),
            const CircularProgressIndicator(),
          ],
        ),
      ),
    );
  }
}

class LoginScreen extends StatelessWidget {
  const LoginScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Card(
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'Welcome',
                    style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 24),
                  TextFormField(
                    initialValue: 'demo.user@example.com',
                    decoration: const InputDecoration(labelText: 'Email'),
                  ),
                  const SizedBox(height: 16),
                  TextFormField(
                    initialValue: 'password123',
                    obscureText: true,
                    decoration: const InputDecoration(labelText: 'Password'),
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(builder: (_) => const HomeScreen()),
                      );
                    },
                    child: const Text('Login'),
                  ),
                  const SizedBox(height: 10),
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(builder: (_) => const HomeScreen()),
                      );
                    },
                    child: const Text('Create account'),
                  )
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final items = [
      _FeatureCard(title: 'Search Spare Part', icon: Icons.search, onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SearchScreen()))),
      _FeatureCard(title: 'Check Availability', icon: Icons.check_circle, onTap: () => _showPredictionAction(context, 'availability')),
      _FeatureCard(title: 'Estimate Price', icon: Icons.attach_money, onTap: () => _showPredictionAction(context, 'price')),
      _FeatureCard(title: 'Estimate Waiting Time', icon: Icons.timer, onTap: () => _showPredictionAction(context, 'wait')),
      _FeatureCard(title: 'Find Garage', icon: Icons.store, onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const GarageScreen()))),
      _FeatureCard(title: 'Service Request', icon: Icons.assignment, onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const ServiceRequestScreen()))),
    ];

    return Scaffold(
      appBar: AppBar(title: const Text('Home Dashboard')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: GridView.count(
          crossAxisCount: 2,
          childAspectRatio: 1.15,
          crossAxisSpacing: 12,
          mainAxisSpacing: 12,
          children: items,
        ),
      ),
    );
  }

  void _showPredictionAction(BuildContext context, String type) {
    final apiService = ApiService();
    final payload = {
      'bike_model': 'Royal Enfield Thunderbird 350X',
      'manufacturing_year': 2019,
      'part_name': 'Clutch Cable',
      'part_category': 'Control',
      'supplier_type': 'Online Supplier',
      'city': 'Pune',
      'part_condition': 'New',
      'service_priority': 'High',
    };

    Future(() async {
      switch (type) {
        case 'availability':
          return await apiService.predictAvailability(payload);
        case 'price':
          return await apiService.predictPrice(payload);
        case 'wait':
          return await apiService.predictWait(payload);
        default:
          return {'message': 'Not supported'};
      }
    }).then((result) {
      if (!context.mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => PredictionResultScreen(result: result, type: type),
        ),
      );
    }).catchError((error) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('API error: ${error.toString()}')),
      );
    });
  }
}

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final _formKey = GlobalKey<FormState>();
  final _bikeModel = TextEditingController(text: 'Royal Enfield Thunderbird 350X');
  final _year = TextEditingController(text: '2019');
  final _partName = TextEditingController(text: 'Clutch Cable');
  final _partCategory = TextEditingController(text: 'Control');
  final _city = TextEditingController(text: 'Pune');
  final _condition = TextEditingController(text: 'New');
  final _priority = TextEditingController(text: 'High');

  bool _loading = false;

  Future<void> _search() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    final apiService = ApiService();
    try {
      final result = await apiService.predictPartDetails({
        'bike_model': _bikeModel.text,
        'manufacturing_year': int.tryParse(_year.text) ?? 2019,
        'part_name': _partName.text,
        'part_category': _partCategory.text,
        'city': _city.text,
        'part_condition': _condition.text,
        'service_priority': _priority.text,
      });
      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => PredictionResultScreen(result: result, type: 'details')),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to fetch prediction: $error')),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Spare Part Search')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              TextFormField(controller: _bikeModel, decoration: const InputDecoration(labelText: 'Bike Model')),
              TextFormField(controller: _year, decoration: const InputDecoration(labelText: 'Manufacturing Year')),
              TextFormField(controller: _partName, decoration: const InputDecoration(labelText: 'Part Name')),
              TextFormField(controller: _partCategory, decoration: const InputDecoration(labelText: 'Part Category')),
              TextFormField(controller: _city, decoration: const InputDecoration(labelText: 'City')),
              TextFormField(controller: _condition, decoration: const InputDecoration(labelText: 'Condition')),
              TextFormField(controller: _priority, decoration: const InputDecoration(labelText: 'Service Priority')),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _loading ? null : _search,
                child: _loading ? const CircularProgressIndicator() : const Text('Check Part Details'),
              )
            ],
          ),
        ),
      ),
    );
  }
}

class PredictionResultScreen extends StatelessWidget {
  final Map<String, dynamic> result;
  final String type;

  const PredictionResultScreen({super.key, required this.result, required this.type});

  @override
  Widget build(BuildContext context) {
    final availability = result['availability'] ?? 'Unavailable';
    final price = result['estimated_price_inr'] ?? result['estimated_price'] ?? '---';
    final wait = result['estimated_wait_days'] ?? '---';

    return Scaffold(
      appBar: AppBar(title: const Text('Prediction Result')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('PART AVAILABILITY', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 18),
                _ResultRow(label: 'Status', value: availability.toString()),
                _ResultRow(label: 'Estimated Price', value: '₹$price'),
                _ResultRow(label: 'Estimated Waiting', value: '$wait days'),
                _ResultRow(label: 'Recommendation', value: 'Prototype estimate from demo data'),
                const SizedBox(height: 22),
                ElevatedButton(
                  onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const GarageScreen())),
                  child: const Text('View Garage Recommendations'),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class GarageScreen extends StatefulWidget {
  const GarageScreen({super.key});

  @override
  State<GarageScreen> createState() => _GarageScreenState();
}

class _GarageScreenState extends State<GarageScreen> {
  bool _loading = false;
  List<dynamic> _garages = [];

  Future<void> _loadGarages() async {
    setState(() => _loading = true);
    try {
      final response = await ApiService().findGarages({
        'part_name': 'Clutch Cable',
        'city': 'Pune',
      });
      setState(() {
        _garages = response['recommendations'] ?? [];
      });
    } catch (error) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Garage fetch failed: $error')),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Garage Recommendations')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: _loading ? null : _loadGarages,
              child: _loading ? const CircularProgressIndicator() : const Text('Load recommendations'),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView.builder(
                itemCount: _garages.length,
                itemBuilder: (context, index) {
                  final item = _garages[index] as Map<String, dynamic>;
                  return Card(
                    child: ListTile(
                      title: Text(item['garage_name'] ?? 'Garage'),
                      subtitle: Text('${item['city']} • ${item['availability']}'),
                      trailing: Text('${item['recommendation_score']}'),
                    ),
                  );
                },
              ),
            )
          ],
        ),
      ),
    );
  }
}

class ServiceRequestScreen extends StatefulWidget {
  const ServiceRequestScreen({super.key});

  @override
  State<ServiceRequestScreen> createState() => _ServiceRequestScreenState();
}

class _ServiceRequestScreenState extends State<ServiceRequestScreen> {
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _mobile = TextEditingController();
  final _bikeModel = TextEditingController(text: 'Royal Enfield Thunderbird 350X');
  final _part = TextEditingController(text: 'Clutch Cable');
  final _garage = TextEditingController(text: 'Garage 001');
  final _priority = TextEditingController(text: 'High');
  final _description = TextEditingController();
  bool _loading = false;

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      final response = await ApiService().submitServiceRequest({
        'name': _name.text,
        'mobile': _mobile.text,
        'bike_model': _bikeModel.text,
        'part_required': _part.text,
        'preferred_garage': _garage.text,
        'service_priority': _priority.text,
        'request_description': _description.text,
      });
      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => RequestStatusScreen(response: response)),
      );
    } catch (error) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Submission failed: $error')),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Service Request')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              TextFormField(controller: _name, decoration: const InputDecoration(labelText: 'Name')),
              TextFormField(controller: _mobile, decoration: const InputDecoration(labelText: 'Mobile / Email')),
              TextFormField(controller: _bikeModel, decoration: const InputDecoration(labelText: 'Bike Model')),
              TextFormField(controller: _part, decoration: const InputDecoration(labelText: 'Part Required')),
              TextFormField(controller: _garage, decoration: const InputDecoration(labelText: 'Preferred Garage')),
              TextFormField(controller: _priority, decoration: const InputDecoration(labelText: 'Service Priority')),
              TextFormField(controller: _description, maxLines: 4, decoration: const InputDecoration(labelText: 'Request Description')),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _loading ? null : _submit,
                child: _loading ? const CircularProgressIndicator() : const Text('Submit Request'),
              )
            ],
          ),
        ),
      ),
    );
  }
}

class RequestStatusScreen extends StatelessWidget {
  final Map<String, dynamic> response;

  const RequestStatusScreen({super.key, required this.response});

  @override
  Widget build(BuildContext context) {
    final request = response['request'] as Map<String, dynamic>? ?? {};
    return Scaffold(
      appBar: AppBar(title: const Text('Request Status')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('Request ID: ${request['request_id'] ?? '---'}', style: const TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 12),
                Text('Name: ${request['name'] ?? '---'}'),
                Text('Bike Model: ${request['bike_model'] ?? '---'}'),
                Text('Part Required: ${request['part_required'] ?? '---'}'),
                Text('Preferred Garage: ${request['preferred_garage'] ?? '---'}'),
                Text('Status: ${request['status'] ?? 'submitted'}'),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ResultRow extends StatelessWidget {
  final String label;
  final String value;

  const _ResultRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
          Text(value),
        ],
      ),
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final VoidCallback onTap;

  const _FeatureCard({required this.title, required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 36, color: Colors.deepOrange),
              const SizedBox(height: 10),
              Text(title, textAlign: TextAlign.center, style: const TextStyle(fontWeight: FontWeight.w600)),
            ],
          ),
        ),
      ),
    );
  }
}
