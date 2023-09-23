package ort

import "fmt"

var (
	ErrExportOrtSdk          = fmt.Errorf("Export Ort Sdk Error")
	ErrZeroShapeLength       = fmt.Errorf("The shape has no dimensions")
	ErrShapeOverflow         = fmt.Errorf("The shape's flattened size overflows an int64")
	ErrShapeNotEqual         = fmt.Errorf("The shape compare not equal")
	ErrBindingLengthNotEqual = fmt.Errorf("The binding data length compare not equal")
)

// This type of error is returned when we attempt to validate a tensor that has
// a negative or 0 dimension.
type BadShapeDimensionError struct {
	DimensionIndex int
	DimensionSize  int64
}

func (e *BadShapeDimensionError) Error() string {
	return fmt.Sprintf("Dimension %d of the shape has invalid value %d",
		e.DimensionIndex, e.DimensionSize)
}

type OrtStatusError struct {
	Msg  string
	Code OrtErrorCode
}

func (s *OrtStatusError) Error() string {
	return fmt.Sprintf("%s (code=%d)", s.Msg, s.Code)
}
