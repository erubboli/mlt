package main

import "github.com/gonum/matrix/mat64"

func Hypothesis(x, theta *mat64.Vector) float64 {
	var res mat64.Dense
	res.Mul(x.T(), theta)
	return res.At(0, 0)
}

func MultiHypothesis(x *mat64.Dense, theta *mat64.Vector) *mat64.Vector {
	var res mat64.Dense
	res.Mul(theta.T(), x)
	return res.RowView(0)
}
