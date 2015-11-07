package main

import (
  "github.com/gonum/matrix/mat64"
  "testing"
)

func TestHypothesis(t *testing.T) {

  theta := mat64.NewVector(2,[]float64{0, 2})
  examples := mat64.NewDense(3, 1, []float64{1,2,10})
  results := mat64.NewVector(3,[]float64{2,4,20})

  for i := 0; i<3; i++ {

    x := mat64.NewVector(1,[]float64{examples.At(i,0)})
    res := results.At(i,0)
    h := Hypothesis(x,theta)

    if h != res {
      t.Errorf("hypothesis(%v) is expected to be equal to %v, found %v", x,res,h)
    }

  }
}
