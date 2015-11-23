package main

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

const EPSILON float64 = 0.00000001

func floatDiffers(a, b float64) bool {
	if (a-b) < EPSILON && (b-a) < EPSILON {
		return false
	}
	return true
}

func TestCostFunction(t *testing.T) {
	for _, test := range []struct {
		x            *mat64.Dense
		y            *mat64.Vector
		theta        *mat64.Vector
		expectedCost float64
	}{
		{
			mat64.NewDense(4, 2, []float64{1, 2, 1, 3, 1, 4, 1, 5}),
			mat64.NewVector(4, []float64{7, 6, 5, 4}),
			mat64.NewVector(2, []float64{0.1, 0.2}),
			11.9450,
		}, {
			mat64.NewDense(4, 3, []float64{1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6}),
			mat64.NewVector(4, []float64{7, 6, 5, 4}),
			mat64.NewVector(3, []float64{0.1, 0.2, 0.3}),
			7.0175,
		},
	} {

		cost := Cost(test.x, test.y, test.theta)

		if floatDiffers(cost, test.expectedCost) {
			t.Errorf("Cost is expected to be equal to %v, found %v", test.expectedCost, cost)
		}
	}
}
