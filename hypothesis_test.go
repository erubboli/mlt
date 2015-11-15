package main

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestHypothesis(t *testing.T) {
	for _, test := range []struct {
		theta *mat64.Vector
		x     *mat64.Vector
		y     float64
	}{
		{
			mat64.NewVector(2, []float64{0, 2}),
			mat64.NewVector(2, []float64{0, 1}),
			2.0,
		}, {
			mat64.NewVector(2, []float64{0, 2}),
			mat64.NewVector(2, []float64{0, 2}),
			4.0,
		}, {
			mat64.NewVector(2, []float64{0, 2}),
			mat64.NewVector(2, []float64{0, 10}),
			20.0,
		}, {
			mat64.NewVector(2, []float64{1, 2}),
			mat64.NewVector(2, []float64{1, 10}),
			21.0,
		}, {
			mat64.NewVector(3, []float64{1, 2.5, 5}),
			mat64.NewVector(3, []float64{10, 20, 0}),
			60.0,
		},
	} {
		h := Hypothesis(test.x, test.theta)

		if h != test.y {
			t.Errorf("Hypothesis(%v,%v) is expected to be equal to %v, found %v", test.x, test.theta, test.y, h)
		}
	}
}

func TestMultiHypothesis(t *testing.T) {
	for _, test := range []struct {
		theta *mat64.Vector
		x     *mat64.Dense
		y     *mat64.Vector
	}{
		{
			mat64.NewVector(2, []float64{0, 2}),
			mat64.NewDense(2, 3, []float64{0, 0, 0, 1, 2, 10}),
			mat64.NewVector(3, []float64{2, 4, 20}),
		},
	} {
		h := MultiHypothesis(test.x, test.theta)

		if !mat64.Equal(h, test.y) {
			t.Errorf("MultiHypothesis(%v,%v) is expected to be equal to %v, found %v", test.x, test.theta, test.y, h)
		}
	}

}
