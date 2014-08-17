package util;


import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class Util {
	public static DoubleMatrix sigmoidrndMat(DoubleMatrix in) {
		return sigmoidMat(in).gt(DoubleMatrix.rand(in.rows, in.columns));
	}

	public static DoubleMatrix sigmoidMat(DoubleMatrix in) {
		return MatrixFunctions.exp(in.neg()).add(1).rdiv(1);

	}
}
