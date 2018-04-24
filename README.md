# Gonet
Package __gonet__ implements a simple fully-connected neural network.
It uses a small matrix package [gomat](http://github.com/acatovic/gomat).
Otherwise it is completely self-contained.

Gonet implements two structs - `Dataset` and `Network`.
`Dataset` is a simple struct used for loading the training and test data.
`Network` uses `Dataset` to fit/train itself, i.e. using backpropagation
to find optimal weight and bias values. It exposes `Fit()` and `Transform()` methods; the former is used for training and the latter for inference.

To install __gonet__ simply run `go get github.com/acatovic/gonet`.

## Examples ##
### Train a network so the input in the range (0, 1) is halved ###
```go
import (
  "fmt"
  "github.com/acatovic/gonet"
)

func main() {
  // Training data; x are input variables and y are labels
  x := [][]float64{{0.2},{0.3},{0.4},{0.5},{0.8}}
  y := [][]float64{{0.1},{0.15},{0.2},{0.25},{0.4}}
  training_data := Dataset(x, y)

  // Create a 3-layer neural network; one neuron at input,
  // four neurons in the hidden layer, and one neuron at output
  layers := []int{1,4,1}
  net := New(layers)

  // Set the number of epochs, our learning rate (eta) and
  // start training
  epochs := 50000
  eta := 3.0
  net.Fit(training_data, epochs, eta, false)

  // Evaluate against some test data
  // NOTE: we're using some completely unseen inputs
  x_test := [][]float64{{0.1},{0.2},{0.6},{0.8}}
  for i := 0; i < len(x_test); i++ {
    fmt.Printf("* Input  (x): %v\n", x_test[i])
    y_test := net.Transform(x_test[i])
    fmt.Printf("  Output (y): %v\n", y_test)
  }
}
```