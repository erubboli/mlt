package main

import "github.com/gonum/matrix/mat64"

func Cost(x *mat64.Dense, y, theta *mat64.Vector) float64 {
	ar, _ := x.Dims()
	_, bc := theta.Dims()
	h := mat64.NewDense(ar, bc, make([]float64, ar*bc))
	squaredErrors := mat64.NewDense(ar, 1, make([]float64, ar))

	h.Mul(x, theta)
	squaredErrors.Sub(h, y)
	squaredErrors.MulElem(squaredErrors, squaredErrors)
	l := 1.0 / (2.0 * float64(ar))
	j := mat64.Sum(squaredErrors) * l
	return j
}
