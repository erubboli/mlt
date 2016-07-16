package main

import (
    "testing"

    "github.com/gonum/matrix/mat64"
)

func TestGradientDescent(t *testing.T) {

    alpha := 0.01
    maxIters := 15000
    tolerance := 0.0001
    for _, test := range []struct {
        x      *mat64.Dense
        y      *mat64.Vector
        result *mat64.Vector
    }{
        {
            mat64.NewDense(3, 4, []float64{
                1, 3, 5, 6,
                1, 1, 2, 3,
                1, 9, 4, 2}),
            mat64.NewVector(3, []float64{1, 6, 4}),
            mat64.NewVector(4, []float64{8.0918, 0.8920, -3.7990, 1.5379}),
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
            mat64.NewVector(4, []float64{0.6665, 1.3335, 2.0000, 2.6665}),
        },
    } {

        theta := GradientDescent(test.x, test.y, alpha, tolerance, maxIters)

        if !mat64.EqualApprox(test.result, theta, 0.0001) {
            t.Error("Expected:", test.result)
            t.Error("Actual:  ", theta)
        }
    }
}
