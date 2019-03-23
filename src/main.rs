extern  crate  rand;
use rand::distributions::{Normal, Distribution};
fn sigmoid(x:f64) -> f64{1.0_f64 / (1.0_f64 + (-x).exp())}
fn rand_norm() -> f64{Normal::new(0.0, 2.0).sample(&mut rand::thread_rng()) }
fn deriv_sigmoid(x:f64) -> f64{sigmoid(x) * (1.0_f64 - sigmoid(x))}
fn feed_forward(input:&Vec<f64>, weights:&Vec<f64>, biases:&Vec<f64>) -> f64{
    let hidden1 = sigmoid(weights[0] * input[0] + weights[1] * input[1] + biases[0]);
    let hidden2 = sigmoid(weights[2] * input[0] + weights[3] * input[1] + biases[1]);
    sigmoid(weights[4] * hidden1 + weights[5] * hidden2 + biases[2])}
fn train(train_inpt:&Vec<Vec<f64>>, train_out:&Vec<f64>, weights:&mut Vec<f64>, biases:&mut Vec<f64>, learning_rate:f64, epochs:i32){
    for _i in 0..epochs{
        for x in 0..train_out.len(){
            let input :&Vec<f64> = &train_inpt[x];
            let sum_hidden1 = weights[0] * input[0] + weights[1] * input[1] + biases[0];
            let hidden1 = sigmoid(sum_hidden1);
            let sum_hidden2 = weights[2] * input[0] + weights[3] * input[1] + biases[1];
            let hidden2= sigmoid(sum_hidden2);
            let sum_output = weights[4] * hidden1 + weights[5] * hidden2 + biases[2];
            let output = sigmoid(sum_output);
            let d_l_d_out = -2_f64 * (train_out[x] - output);
            let d_out_d_w4 = hidden1 * deriv_sigmoid(sum_output);
            let d_out_d_w5 = hidden2 * deriv_sigmoid(sum_output);
            let d_out_d_b2 = deriv_sigmoid(sum_output);
            let d_out_d_h1 = weights[4] * deriv_sigmoid(sum_output);
            let d_out_d_h2 = weights[5] * deriv_sigmoid(sum_output);
            let d_h1_d_w0 = input[0] * deriv_sigmoid(sum_hidden1);
            let d_h1_d_w1 = input[1] * deriv_sigmoid(sum_hidden1);
            let d_h1_d_b0 = deriv_sigmoid(sum_hidden1);
            let d_h2_d_w2 = input[0] * deriv_sigmoid(sum_hidden2);
            let d_h2_d_w3 = input[1] * deriv_sigmoid(sum_hidden2);
            let d_h2_d_b1 = deriv_sigmoid(sum_hidden2);
            weights[0] -= learning_rate * d_l_d_out * d_out_d_h1 * d_h1_d_w0;
            weights[1] -= learning_rate * d_l_d_out * d_out_d_h1 * d_h1_d_w1;
            biases[0] -= learning_rate * d_l_d_out * d_out_d_h1 * d_h1_d_b0;
            weights[2] -= learning_rate * d_l_d_out * d_out_d_h2 * d_h2_d_w2;
            weights[3] -= learning_rate * d_l_d_out * d_out_d_h2 * d_h2_d_w3;
            biases[1] -= learning_rate * d_l_d_out * d_out_d_h2 * d_h2_d_b1;
            weights[4] -= learning_rate * d_l_d_out * d_out_d_w4;
            weights[5] -= learning_rate * d_l_d_out * d_out_d_w5;
            biases[2] -= learning_rate * d_l_d_out * d_out_d_b2;}}}

fn main() {
    let mut weights = vec![rand_norm(), rand_norm(), rand_norm(), rand_norm(), rand_norm(), rand_norm()];
    let mut biases = vec![rand_norm(), rand_norm(), rand_norm()];
    let train_inpt = vec!(vec![0_f64, 0_f64],vec![0_f64, 1_f64],vec![1_f64, 0_f64],vec![1_f64, 1_f64]);
    let train_out = vec!(0_f64,1_f64,1_f64,0_f64); //your training outputs
    let test_in = vec![0_f64, 0_f64]; //your test values
    train(&train_inpt, &train_out, &mut weights, &mut biases, 0.3, 200);
    println!("\nResult:{}",feed_forward(&test_in, &weights, &biases).round());
}