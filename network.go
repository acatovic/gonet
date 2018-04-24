// Copyright 2018 Armin Catovic. All rights reserved.
// Use of this source code is governed by a MIT license
// that can be found in the LICENSE file.

package gonet

import (
	"fmt"
	"github.com/acatovic/gomat"
)

type Network struct {
	num_layers int
	biases []*gomat.Matrix
	weights []*gomat.Matrix
}

// "CONSTRUCTORS"

// New is a Network "constructor"; it ensures network
// is properly initialised, returning a ptr to the
// initialised Network struct
func New(layers []int) *Network {
	nw := &Network{num_layers: len(layers)}
	nw.biases = make([]*gomat.Matrix, nw.num_layers-1)
	nw.weights = make([]*gomat.Matrix, nw.num_layers-1)

	for i := 1; i < nw.num_layers; i++ {
		nw.biases[i-1] = gomat.Randn(layers[i], 1)
	}

	for i, j := 0, 1; i < nw.num_layers-1; i, j = i+1, j+1 {
		nw.weights[i] = gomat.Randn(layers[j], layers[i])
	}

	return nw
}

// PUBLIC METHODS

// Fit is the training function, i.e. it runs over
// the dataset for a number of epochs and updates
// the networks weights and biases
func (net *Network) Fit(train_data *dataset, epochs int, eta float64,
	verbose bool) {
	for i := 0; i < epochs; i++ {
		train_data.Shuffle()
		for j := 0; j < train_data.Size(); j++ {
			as, zs := net.feedforward(train_data.x[j])
			if verbose {
				fmt.Println("Epoch: ", i, "Input: ",
					train_data.x[j].ToVec(),
					"Output: ", as[len(as) - 1].ToVec())
			}
			dbiases, dweights := net.backprop(train_data.y[j], as, zs)
			net.update(eta, dbiases, dweights)
		}
	}
}

// Transform is the inference function, taking vector x
// as input and producing predicted output vector y
func (net *Network) Transform(x []float64) []float64 {
	x_mat := gomat.FromVec(x)
	activations, _ := net.feedforward(x_mat)
	y_mat := activations[len(activations) - 1]
	return y_mat.ToVec()
}

// PRIVATE METHODS

// backprop executes the cost function, takes its
// derivative, then back-propagates the cost derivative
// through the network, calculating weight and bias
// deltas
// y is a ptr to expected/target output
// as is a vector of activation outputs
// zs is a vector of z-vectors (inputs to activation funcs)
// Returns (bias, weight) derivatives for each layer
func (net *Network) backprop(y *gomat.Matrix, as, zs []*gomat.Matrix) (
	[]*gomat.Matrix, []*gomat.Matrix) {
	dbiases := make([]*gomat.Matrix, net.num_layers - 1)
	dweights := make([]*gomat.Matrix, net.num_layers - 1)
	delta := gomat.Mul(net.cost_derivative(as[len(as) - 1], y),
		gomat.Sigmoidpr(zs[len(zs) - 1]))

	// output layer partial derivatives
	dbiases[len(dbiases) - 1] = delta
	dweights[len(dweights) - 1] = gomat.Dot(delta,
		gomat.Transpose(as[len(as) - 2]))
	
	// hidden layer partial derivatives
	for i := 2; i < net.num_layers; i++ {
		z := zs[len(zs) - i]
		sp := gomat.Sigmoidpr(z)
		wi := len(net.weights) - i + 1
		delta = gomat.Mul(gomat.Dot(
			gomat.Transpose(net.weights[wi]), delta), sp)
		dbiases[len(dbiases) - i] = delta
		ai := len(as) - i - 1
		dweights[len(dweights) - i] = gomat.Dot(delta,
			gomat.Transpose(as[ai]))
	}

	return dbiases, dweights
}

// cost_derivative returns a partial derivative of
// quadtratic cost function
func (net *Network) cost_derivative(out, y *gomat.Matrix) *gomat.Matrix {
	return gomat.Sub(out, y)
}

// feedforward runs the input x through the network
// returns (activations, z-vectors)
func (net *Network) feedforward(x *gomat.Matrix) (
	[]*gomat.Matrix, []*gomat.Matrix) {
	activations := make([]*gomat.Matrix, net.num_layers)
	zvectors := make([]*gomat.Matrix, net.num_layers - 1)
	a := x
	activations[0] = a
	for i := 0; i < len(net.weights); i++ {
		z := gomat.Add(gomat.Dot(net.weights[i], a), net.biases[i])
		zvectors[i] = z
		a = gomat.Sigmoid(z)
		activations[i + 1] = a
	}

	return activations, zvectors
}

// update updates network biases and weights using the results
// from backpropagation
func (net *Network) update(eta float64, dbiases, dweights []*gomat.Matrix) {
	for i := 0; i < len(net.weights); i++ {
		net.biases[i] = gomat.Sub(net.biases[i],
			gomat.Scale(eta, dbiases[i]))
		net.weights[i] = gomat.Sub(net.weights[i],
			gomat.Scale(eta, dweights[i]))
	}
}