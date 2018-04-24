// Copyright 2018 Armin Catovic. All rights reserved.
// Use of this source code is governed by a MIT license
// that can be found in the LICENSE file.

// Package gonet implements a simple fully-connected
// neural network. It uses a small matrix package
// from github.com/acatovic/gomat. Otherwise it is
// completely self-contained.
//
// Gonet provides a base for further work in the area
// of artificial neural networks in Go.
//
// Gonet implements two structs - Dataset and Network.
// Dataset is a simple struct used for loading training
// and test data.
// Network uses Dataset to fit/train itself, i.e. using
// backpropagation to find optimal weight and bias values.
// It exposes Fit() and Transform() methods; the former
// is used for training and the latter for inference.
//
// Example - Train a network so the input is halved;
// the input is a floating point value between 0 and 1.
//  x := [][]float64{{0.2},{0.3},{0.4},{0.5},{0.8}}
//  y := [][]float64{{0.1},{0.15},{0.2},{0.25},{0.4}}
//  training_data := Dataset(x, y)
//  layers := []int{1,4,1}
//  net := New(layers)
//  epochs := 50000
//  eta := 3.0
//  net.Fit(training_data, epochs, eta, false)
//  x_test := [][]float64{{0.1},{0.2},{0.6},{0.8}}
//  for i := 0; i < len(x_test); i++ {
//      fmt.Printf("      * Input  (x): %v\n", x_test[i])
//      y_test := net.Transform(x_test[i])
//      fmt.Printf("        Output (y): %v\n", y_test)
//  }
package gonet