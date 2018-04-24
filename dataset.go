// Copyright 2018 Armin Catovic. All rights reserved.
// Use of this source code is governed by a MIT license
// that can be found in the LICENSE file.

package gonet

import (
	"github.com/acatovic/gomat"
	"math/rand"
	"time"
)

type dataset struct {
	x []*gomat.Matrix
	y []*gomat.Matrix
}

// "CONSTRUCTORS"

// Dataset is a "constructor", ensuring training data
// is properly loaded into the dataset struct.
// x is a vector of input vectors (feature values), and
// y is a vector of target output vectors (labels), i.e.
// x[0] will hold the feature values for the first sample,
// and y[0] will hold a corresponding target output/label.
func Dataset(x, y [][]float64) *dataset {
	if len(x) != len(y) {
		panic("X and Y must be same length")
	}
	dat := &dataset{make([]*gomat.Matrix, len(x)),
		make([]*gomat.Matrix, len(y))}
	for i := 0; i < len(x); i++ {
		dat.x[i] = gomat.FromVec(x[i])
		dat.y[i] = gomat.FromVec(y[i])
	}
	return dat
}

// PUBLIC METHODS

// Shuffle implements Fisher-Yates shuffle,
// randomising data
func (dat *dataset) Shuffle() {
	rand.Seed(time.Now().UnixNano())
	for i := range dat.x {
		j := rand.Intn(i + 1)
		dat.x[i], dat.x[j] = dat.x[j], dat.x[i]
		dat.y[i], dat.y[j] = dat.y[j], dat.y[i]
	}
}

func (dat *dataset) Size() int {
	return len(dat.x)
}