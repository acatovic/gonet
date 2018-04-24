// Copyright 2018 Armin Catovic. All rights reserved.
// Use of this source code is governed by a MIT license
// that can be found in the LICENSE file.

package gonet

import (
	"fmt"
	"testing"
)

func binary_rounding(x []float64) []int {
	x_int := make([]int, len(x))
	for i := range x {
		if x[i] >= 0.9 {
			x_int[i] = 1
		} else {
			x[i] = 0
		}
	}
	return x_int
}

// Halves any value between 0 and 1
func TestNetwork_Half(*testing.T) {
	x := [][]float64{{0.2},{0.3},{0.4},{0.5},{0.8}}
	y := [][]float64{{0.1},{0.15},{0.2},{0.25},{0.4}}
	training_data := Dataset(x, y)
	layers := []int{1,4,1}
	net := New(layers)
	epochs := 50000
	eta := 3.0

	// Training
	fmt.Printf("INFO: Training Started\n")
	net.Fit(training_data, epochs, eta, false)
	fmt.Printf("INFO: Training Completed\n")

	// Inference
	x_test := [][]float64{{0.1},{0.2},{0.6},{0.8}}
	fmt.Printf("INFO: Testing Started\n")
	for i := 0; i < len(x_test); i++ {
		fmt.Printf("      * Input  (x): %v\n", x_test[i])
		y_test := net.Transform(x_test[i])
		fmt.Printf("        Output (y): %v\n", y_test)
	}
}

// Trains a 3-bit binary counter
func TestNetwork_BinaryCounter(*testing.T) {
	x := [][]float64{{0,0,0},
                     {0,0,1},
                     {0,1,0},
                     {0,1,1},
                     {1,0,0},
					 {1,0,1},
					 {1,1,0},
                     {1,1,1}}
	y := [][]float64{{0,0,1},
                     {0,1,0},
                     {0,1,1},
                     {1,0,0},
                     {1,0,1},
					 {1,1,0},
					 {1,1,1},
                     {0,0,0}}
	
	// Training
	training_data := Dataset(x, y)
	layers := []int{3,6,3}
	net := New(layers)
	epochs := 150
	eta := 3.0
	fmt.Printf("INFO: Training Started\n")
	net.Fit(training_data, epochs, eta, true)
	fmt.Printf("INFO: Training Completed\n")

	// Inference
	x_test := [][]float64{{1,0,0},
						  {1,1,1},
						  {0,0,0}}
	fmt.Printf("INFO: Testing Started\n")
	for i := 0; i < len(x_test); i++ {
		fmt.Printf("      * Input  (x): %v\n", x_test[i])
		y_test := binary_rounding(net.Transform(x_test[i]))
		fmt.Printf("        Output (y): %v\n", y_test)
	}
}