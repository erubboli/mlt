package main

import (
  "github.com/gonum/matrix/mat64"
)

func Hypothesis(x, theta *mat64.Vector) float64{
  var res mat64.Dense
  transposed := theta.T()
  res.Mul(x, transposed)

  return res.At(0,1)
}

