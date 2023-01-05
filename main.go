package main

import (
	"github.com/ThomasCardin/nnfs/pkg/batches"
	"gonum.org/v1/gonum/mat"
)

func main() {
	X := mat.NewDense(3, 4, []float64{1, 2, 3, 2.5, 2.0, 5.0, -1.0, 2.0, -1.5, 2.7, 3.3, -0.8}) // "data set"

	layerDense1 := batches.CreateLayerDense(4, 5)
	layerDense1.Forward(X)
	layerDense1.ToString()

	layerDense2 := batches.CreateLayerDense(5, 2)
	layerDense2.Forward(&layerDense1.Output)
	layerDense2.ToString()
}