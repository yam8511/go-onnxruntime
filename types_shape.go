package ort

import "fmt"

type Shape []int64

func NewShape(dimensions ...int64) Shape {
	return Shape(dimensions)
}

func (s Shape) Sizes() []int {
	sizes := []int{}
	for _, v := range s {
		sizes = append(sizes, int(v))
	}
	return sizes
}

func (s Shape) FlattenedSize() int64 {
	if len(s) == 0 {
		return 0
	}
	toReturn := int64(s[0])
	for i := 1; i < len(s); i++ {
		toReturn *= s[i]
	}
	return toReturn
}

func (s Shape) Validate() error {
	if len(s) == 0 {
		return ErrZeroShapeLength
	}
	if s[0] <= 0 {
		return &BadShapeDimensionError{
			DimensionIndex: 0,
			DimensionSize:  s[0],
		}
	}
	flattenedSize := int64(s[0])
	for i := 1; i < len(s); i++ {
		d := s[i]
		if d <= 0 {
			return &BadShapeDimensionError{
				DimensionIndex: i,
				DimensionSize:  d,
			}
		}
		tmp := flattenedSize * d
		if tmp < flattenedSize {
			return ErrShapeOverflow
		}
		flattenedSize = tmp
	}
	return nil
}

// Makes and returns a deep copy of the Shape.
func (s Shape) Clone() Shape {
	toReturn := make([]int64, len(s))
	copy(toReturn, []int64(s))
	return Shape(toReturn)
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int64(s))
}

// Returns true if both shapes match in every dimension.
func (s Shape) Equals(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] != other[i] {
			return false
		}
	}
	return true
}
