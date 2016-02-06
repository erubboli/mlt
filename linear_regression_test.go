package main

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestLinearRegression(t *testing.T) {
	t.Skip("Skipping for now")
	for _, test := range []struct {
		x          *mat64.Dense
		y          *mat64.Vector
		result     *mat64.Vector
		test       *mat64.Vector
		testResult float64
	}{
		{
			mat64.NewDense(3, 4, []float64{
				1, 3, 5, 6,
				1, 1, 2, 3,
				1, 9, 4, 2}),
			mat64.NewVector(3, []float64{1, 6, 4}),
			mat64.NewVector(4, []float64{8.0918, 0.8920, -3.7990, 1.5379}),
			mat64.NewVector(4, []float64{1, 1, 2, 3}),
			6.0,
		}, {
			mat64.NewDense(10, 4, []float64{
				1, 2, 3, 4,
				1, 3, 4, 5,
				1, 4, 5, 6,
				1, 5, 6, 7,
				1, 6, 7, 8,
				1, 7, 8, 9,
				1, 8, 9, 10,
				1, 9, 10, 11,
				1, 10, 11, 12,
				1, 11, 12, 13}),
			mat64.NewVector(10, []float64{20, 26, 32, 38, 44, 50, 56, 62, 68, 74}),
			mat64.NewVector(4, []float64{0, 1, 2, 3}),
			mat64.NewVector(4, []float64{1, 10, 11, 12}),
			68.0,
		},
	} {

		lr := NewLinearRegression(test.x, test.y)
		lr.Fit()
		if !mat64.EqualApprox(test.result, lr.Theta, 0.0001) {
			t.Errorf("LinearRegressions's return theta is expected to be equal to %v, found %v", test.result, lr.Theta)
		}
		predicted := lr.Predict(test.test)

		if math.Abs(test.testResult-predicted) > 0.0001 {
			t.Errorf("LinearRegression predict values are expected to be equal to %f, found %f", test.testResult, predicted)
		}

	}

}
