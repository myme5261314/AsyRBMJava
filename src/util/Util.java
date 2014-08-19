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

	public static DoubleMatrix tanh_optMat(DoubleMatrix in) {
		return MatrixFunctions.tanh(in.mul(2.0d / 3)).mul(1.7159);
	}

	// The start and end is all inclusive.
	public static int[] generateIdx(int start, int end) {
		assert (start >= 0);
		assert (end >= start);
		int[] out = new int[end - start + 1];
		for (int i = 0; i < out.length; i++) {
			out[i] = start + i;
		}
		return out;
	}

}
