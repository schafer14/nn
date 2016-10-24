extern crate rand;

use std::f64;

const LEARNING_RATE: f64 = 1f64;

struct Network {
	layers: Vec<Layer>
}

struct Layer {
	neurons: Vec<Neuron>,
}

struct Neuron {
	weights: Vec<f64>,
	output: Option<f64>,
	inputs: Vec<f64>
}

impl Network {
	fn new (number_of_features: usize, layers_sizes: Vec<usize>) -> Network {
		let mut layers: Vec<Layer> = Vec::new();

		for (i, size) in layers_sizes.iter().enumerate() {
			let number_of_weights: usize = if i == 0 { number_of_features } else { layers_sizes[i - 1] };
			let layer = Layer::new(*size, number_of_weights + 1);
			layers.push(layer);
		}

		Network { layers: layers }
	}

	fn feed_forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
		let mut result = Vec::clone(inputs);

		for i in 0..self.layers.len() {
			// Bias input
			result.push(1f64);

			let ref mut layer: Layer = self.layers[i];

			result = layer.calc(&result);
		}

		result
	}

	#[allow(dead_code)]
	fn total_error(&mut self, inputs: &Vec<f64>, actual: &Vec<f64>) -> f64{
		let result = self.feed_forward(inputs);

		let error_proxy: f64 = result.iter().zip(actual.iter()).map(|(x, y)| (x - y)*(x - y)).sum();

		error_proxy / 2f64
	}

	fn online_back_propogation(&mut self, inputs: &Vec<f64>, actual: &Vec<f64>) {
		self.feed_forward(inputs);

		let index_of_last_layer: usize = self.layers.len() - 1;
		let mut d_errors_wrt_input: Vec<Vec<f64>> = Vec::new();

		// Output layer calc
		let mut d_error_wrt_input_last_layer_vec: Vec<f64> = Vec::new();
		for i in 0..self.layers[index_of_last_layer].neurons.len() {
			let ref neuron: Neuron = self.layers[index_of_last_layer].neurons[i];
			let ref output: f64 = neuron.output.expect("Neuron has no outputted value");

			let d_error_wrt_input: f64 = (output - actual[i]) * output * (1f64 - output);
			d_error_wrt_input_last_layer_vec.push(d_error_wrt_input);
		}

		d_errors_wrt_input.insert(0, d_error_wrt_input_last_layer_vec);

		// Hidden layer calc
		for l in 0..index_of_last_layer {
			let mut d_errors_wrt_inputs_hidden_layer_vec: Vec<f64> = Vec::new();

			for i in 0..self.layers[l].neurons.len() {
				let ref neuron: Neuron = self.layers[l].neurons[i];
				let ref output: f64 = neuron.output.expect("Neuron has no outputted value");
				let ref d_error_wrt_input_prev = d_errors_wrt_input[0];

				// Each neuron on the latter layer
				let mut d_error_wrt_output: f64 = 0f64;
				for j in 0..self.layers[l + 1].neurons.len() {
					let ref weight: f64 = self.layers[l + 1].neurons[j].weights[i];
					d_error_wrt_output = d_error_wrt_output + d_error_wrt_input_prev[j] * weight;
				}

				let d_error_wrt_input:f64 = d_error_wrt_output * (output * (1f64 * output));
				d_errors_wrt_inputs_hidden_layer_vec.push(d_error_wrt_input);
			}

			d_errors_wrt_input.insert(0, d_errors_wrt_inputs_hidden_layer_vec);
		}

		// Update output layer
		for l in 0..self.layers.len() {
			for i in 0..self.layers[l].neurons.len() {
				let ref mut neuron: Neuron = self.layers[l].neurons[i];
				
				for j in 0..neuron.weights.len() {
					let d_error_wrt_weight = d_errors_wrt_input[l][i] * neuron.inputs[j];

					neuron.weights[j] = neuron.weights[j] - LEARNING_RATE * d_error_wrt_weight;
				}
			}
		}
	}

}

impl Layer {
	fn new(number_of_neurons: usize, number_of_neuron_on_previous_layer: usize) -> Layer {
		let mut neurons: Vec<Neuron> = Vec::new();
		
		for _ in 0..number_of_neurons {
			let neuron: Neuron = Neuron::new(number_of_neuron_on_previous_layer);
			neurons.push(neuron);
		}

		Layer { neurons: neurons }
	}

	fn calc(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
		let mut results: Vec<f64> = Vec::new();

		for i in 0..self.neurons.len() {
			let ref mut neuron: Neuron = self.neurons[i];
			let result: f64 = neuron.calc(inputs);
			results.push(result);
		}

		results	
	}
}

impl Neuron {
	fn new(number_of_weights: usize) -> Neuron {
		let mut weights: Vec<f64> = Vec::new();

		for _ in 0..number_of_weights {
			let num: f64 = (rand::random::<f64>() - 0.5f64) * 2f64;
			weights.push(num);
		}
		Neuron { weights: weights, output: None, inputs: Vec::new() }
	}

	fn calc(&mut self, inputs: &Vec<f64>) -> f64 {
		let sum: f64 = self.weights.iter().zip(inputs.iter()).map(|(x, y)| x*y).sum();
		let e_to_sum: f64 = (-sum).exp();
		let result = 1f64 / (e_to_sum + 1f64);

		self.output = Some(result);
		self.inputs = Vec::clone(inputs);

		result
	}

}

fn main() {
	let mut nn = Network::new(2, vec![2, 4]);

	let input = vec![0.4f64, 0.36f64];
	let target = vec![0.91f64, 0.25f64, 0.55f64, 0.11f64];

	let e1 = nn.feed_forward(&input);

	for _ in 0..1000000 {
		nn.online_back_propogation(&input, &target);
	}

	let e2 = nn.feed_forward(&input);

	println!("E1 {:?}, e2 {:?}", e1, e2);
}

