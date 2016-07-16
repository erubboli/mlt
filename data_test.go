package main

import (
    "fmt"
    "os"
    "testing"

    "github.com/gonum/matrix/mat64"
)

func TestDataContainer(t *testing.T) {

    file, _ := os.Open("fixtures/2.csv")

    data := NewData(file, "target")
    if data.Cols != 4 {
        t.Error("Expected:", 4)
        t.Error("Actual:  ", data.Cols)
    }

    if data.Rows != 3 {
        t.Error("Expected:", 3)
        t.Error("Actual:  ", data.Rows)
    }

    if data.Labels[0] != "id" {
        t.Error("Expected:", "id")
        t.Error("Actual:  ", data.Labels[0])
    }

    if data.Data.At(2, 3) != 0.001 {
        t.Error("Expected:", "0.001")
        t.Error("Actual:  ", data.Data.At(2, 3))
    }

    fmt.Printf("data: %0.4v\n", mat64.Formatted(data.Data, mat64.Prefix("      ")))

}
