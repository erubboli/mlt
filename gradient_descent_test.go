package main

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestGradientDescent(t *testing.T) {

	alpha := 0.01
	numIters := 1000
	for _, test := range []struct {
		x      *mat64.Dense
		y      *mat64.Vector
		theta  *mat64.Vector
		result *mat64.Vector
	}{
		{
			mat64.NewDense(3, 3, []float64{3, 5, 6, 1, 2, 3, 9, 4, 2}),
			mat64.NewVector(3, []float64{1, 6, 4}),
			mat64.NewVector(3, []float64{0, 0, 0}),
			mat64.NewVector(3, []float64{1.2123, -2.9458, 2.3219}),
		},
	} {

		new_theta := GradientDescent(test.x, test.y, test.theta, alpha, numIters)

		if !mat64.EqualApprox(test.result, new_theta, 0.0001) {
			t.Errorf("GradientDescent's return theta is expected to be equal to %v, found %v", test.result, new_theta)
		}
	}
}
