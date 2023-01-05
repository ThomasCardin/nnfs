package batches

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

type LayerDense struct {
	Weights mat.Dense
	Biases  mat.Dense
	Output  mat.Dense
}

func CreateLayerDense(nbInputs, nbNeurons int) *LayerDense {
	rand.Seed(0)
	data := make([]float64, nbInputs*nbNeurons)
	for i := range data {
		data[i] = 0.10 * rand.NormFloat64() // between -1 and 1
	}

	return &LayerDense{
		Weights: *mat.NewDense(nbInputs, nbNeurons, data),
		Biases:  *mat.NewDense(1, nbNeurons, nil), // no biases for now
	}
}

func (layerDense *LayerDense) Forward(input *mat.Dense) {
	var result mat.Dense
	result.Mul(input, &layerDense.Weights)
	layerDense.Output = *mat.NewDense(result.RawMatrix().Rows, result.RawMatrix().Cols, nil)
	for i := 0; i < result.RawMatrix().Rows; i++ {
		for j := 0; j < result.RawMatrix().Cols; j++ {
			output := result.At(i, j) + layerDense.Biases.At(0, j)
			layerDense.Output.Set(i, j, output)
		}
	}
}

func (layerDense *LayerDense) ToString() {
	fmt.Println("==== LAYER ====")
	for i := 0; i < layerDense.Output.RawMatrix().Rows; i++ {
		row := []float64{}
		for j := 0; j < layerDense.Output.RawMatrix().Cols; j++ {
			row = append(row, layerDense.Output.At(i, j))
		}
		fmt.Println(row)
	}
	fmt.Println("===============")
}
