extern crate rand;

use std::f64;
use std::cell::Cell;

const learning_rate: f64 = 1f64;

struct Network {
	layers: Vec<Layer>
}

struct Layer {
	neurons: Vec<Neuron>
}

struct Neuron {
	weights: Vec<f64>,
	output: Cell<Option<f64>>
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

	fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
		let mut result = Vec::clone(inputs);

		for layer in self.layers.iter() {
			result.push(1f64);

			result = layer.calc(&result);
		}

		result
	}

	fn total_error(&mut self, inputs: &Vec<f64>, actual: &Vec<f64>) -> f64{
		let result = self.feed_forward(inputs);

		let error_proxy: f64 = result.iter().zip(inputs.iter()).map(|(x, y)| (x - y)*(x - y)).sum();

		error_proxy / 2f64
	}

	fn online_back_propogation(&mut self, inputs: &Vec<f64>, actual: &Vec<f64>) {
		let mut result: Vec<f64> = self.feed_forward(inputs);
		let ref mut layers: Layer = self.layers[1];

		for n_i in 0..layers.neurons.len() {
			let ref mut neuron: Neuron = layers.neurons[n_i];
			let ref output: f64 = neuron.output.get().expect("Neuron has no outputted value");
			let mut sum: f64 = 0f64;
			let mut new_errors: Vec<f64> = Vec::new();

			for i in 0..result.len() {
				let d_error_wrt_output: f64 = result[i] - actual[i];
				let d_out_wrt_input: f64 = result[i] * (1f64 - result[i]);
				let d_input_wrt_weight = output;

				let d_error_wrt_weight: f64 = d_error_wrt_output * d_out_wrt_input * d_input_wrt_weight;

				new_errors.push(d_error_wrt_weight);
				sum =  sum + d_error_wrt_weight;
			}

			result = new_errors;
			neuron.weights[0] = neuron.weights[0] - (sum * learning_rate);
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

	fn calc(&self, inputs: &Vec<f64>) -> Vec<f64> {
		self.neurons
			.iter()
			.map(|n| { 
				n.calc(inputs)
			})
			.collect::<Vec<f64>>()
	}
}

impl Neuron {
	fn new(number_of_weights: usize) -> Neuron {
		let mut weights: Vec<f64> = Vec::new();

		for _ in 0..number_of_weights {
			let num: f64 = (rand::random::<f64>() - 0.5f64) * 2f64;
			weights.push(num);
		}
		Neuron { weights: weights, output: Cell::new(None) }
	}

	fn calc(&self, inputs: &Vec<f64>) -> f64 {
		let sum: f64 = self.weights.iter().zip(inputs.iter()).map(|(x, y)| x*y).sum();
		let e_to_sum: f64 = (-sum).exp();
		let result = 1f64 / (e_to_sum + 1f64);
		
		self.output.set(Some(result));

		result
	}

}

fn main() {
	let mut nn = Network::new(2, vec![2, 2]);

	let input = vec![0.4f64, 0.36f64];
	let target = vec![0.91f64, 0.25f64];

	let e1 = nn.feed_forward(&input);

	for _ in 0..100000 {
		let actual = nn.online_back_propogation(&input, &target);
	}

	let e2 = nn.feed_forward(&input);

	println!("E1 {:?}, e2 {:?}", e1, e2)
}

