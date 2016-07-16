package main

import (
    "encoding/csv"
    "fmt"
    "io"
    "strconv"

    "github.com/gonum/matrix/mat64"
)

// Cols are features
// Rows are examples
type Data struct {
    Cols, Rows int          /* Number of features and examples*/
    Labels     []string     /* Names of features */
    Data       *mat64.Dense /* Actual Data */
}

func (d *Data) setFeatures(record []string) {
    d.Cols = len(record)
    d.Labels = record
    d.initializeMatrix()
}

func (d *Data) initializeMatrix() {
    d.Data = mat64.NewDense(d.Rows, d.Cols, nil)
}

func (d *Data) readHeader(r csv.Reader) {
    record, err := r.Read()

    if err != nil {
        fmt.Println("ERROR: ", err)
    }

    d.setFeatures(record)
}

func (d *Data) readFeatures(in csv.Reader) {
    for {
        record, err := in.Read()

        if err == io.EOF {
            break
        }

        if err != nil {
            fmt.Println("ERROR: ", err)
        }

        x := len(record)
        conv := make([]float64, x, x)
        correctData := true

        for i := 0; i < x; i++ {
            conv[i], err = strconv.ParseFloat(record[i], 64)
            if err != nil {
                fmt.Println("ERROR: ", err)
                correctData = false
            }
        }
        if correctData {
            d.AppendRow(conv)
        }
    }
}

func (d *Data) AppendRow(newRow []float64) {
    d.Data = mat64.DenseCopyOf(d.Data.Grow(1, 0))
    d.Rows, d.Cols = d.Data.Dims()
    d.Data.SetRow(d.Rows-1, newRow)
    return
}

func (d *Data) readCSV(in io.Reader) {
    r := csv.NewReader(in)
    d.readHeader(*r)
    d.readFeatures(*r)
}

func NewData(in io.Reader, target string) *Data {
    data := Data{Data: &mat64.Dense{}}
    data.readCSV(in)
    return &data
}
