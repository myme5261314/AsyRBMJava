package DBN;

import java.util.ArrayList;
import java.util.Collections;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;
import org.jblas.util.Permutations;

import util.Option;
import util.Util;

/**
 * The RBM Class
 * 
 * @author peng
 * @param size
 *            int[2], {visible neuron, hidden neuron}
 * @param alpha
 *            double, the learning rate
 * @param momentum
 *            double, the momentum
 * @param W
 *            DoubleMatrix, the weight matrix(visible*hidden)
 * @param vW
 *            DoubleMatrix, the velocity of W's change
 * @param b
 *            DoubleMatrix, the column vector of visible bias
 * @param vb
 *            DoubleMatrix, the velocity of b's change
 * @param c
 *            DoubleMatrix, the column vector of hidden bias
 * @param vc
 *            DoubleMatrix, the velocity of c's change
 * 
 */
public class RBM {
	public int[] size = { 1, 1 };
	public double alpha;
	public double momentum;
	public DoubleMatrix W, vW, b, vb, c, vc;

	public RBM(int[] sz, double alpha, double momentum) {
		assert (sz.length == 2);
		size = sz.clone();
		this.alpha = alpha;
		this.momentum = momentum;
	}

	public boolean setup() {
		W = DoubleMatrix.zeros(size[1], size[0]);
		vW = DoubleMatrix.zeros(size[1], size[0]);
		b = DoubleMatrix.zeros(size[0], 1);
		vb = DoubleMatrix.zeros(size[0], 1);
		c = DoubleMatrix.zeros(size[1], 1);
		vc = DoubleMatrix.zeros(size[1], 1);
		return true;
	}

	public void train(DoubleMatrix x, Option option) {
		int m = x.rows;
		int numbatches = m / option.batchsize;
		for (int i = 0; i < option.numepochs; i++) {
			int[] index = Permutations.randomPermutation(m);
			double error = 0;
			ArrayList<Double> error_list = new ArrayList<Double>();
			for (int j = 0; j < numbatches; j++) {
				int[] oneIndex = new int[option.batchsize];
				System.arraycopy(index, i*option.batchsize, oneIndex, 0, option.batchsize);
				DoubleMatrix batch = x.getRows(oneIndex);
				DoubleMatrix[] grad = computeGradient(batch, option);
				vW.muli(momentum).addi(grad[0]);
				vb.muli(momentum).addi(grad[1]);
				vc.muli(momentum).addi(grad[2]);
				W.addi(vW);
				b.addi(vb);
				c.addi(vc);
				error += grad[3].scalar();
				error_list.add(grad[3].scalar());
			}
			System.out.println("epoch " + (i + 1) + "/" + option.numepochs
					+ ". Average Reconstruction Error is: "
					+ (error / numbatches));
			System.out.println(error_list.toString());
		}
	}

	public DoubleMatrix rbmUp(DoubleMatrix in) {
		return Util.sigmoidMat(in.mmul(W.transpose()).add(
				c.transpose().repmat(in.rows, 1)));
	}

	public DoubleMatrix rbmDown(DoubleMatrix in) {
		return Util
				.sigmoidMat(in.mmul(W).add(b.transpose().repmat(in.rows, 1)));
	}

	private DoubleMatrix[] computeGradient(DoubleMatrix batch, Option option) {
		DoubleMatrix[] out = new DoubleMatrix[4];

		DoubleMatrix t_WT = W.transpose();

		DoubleMatrix v1 = batch;
		DoubleMatrix h1 = Util.sigmoidrndMat(v1.mmul(t_WT).add(
				c.transpose().repmat(option.batchsize, 1)));
		DoubleMatrix v2 = Util.sigmoidrndMat(h1.mmul(W).add(
				b.transpose().repmat(option.batchsize, 1)));
		DoubleMatrix h2 = Util.sigmoidMat(v2.mmul(t_WT).add(
				c.transpose().repmat(option.batchsize, 1)));
		DoubleMatrix c1 = h1.transpose().mmul(v1);
		DoubleMatrix c2 = h2.transpose().mmul(v2);
		out[0] = c1.sub(c2).mul(alpha / option.batchsize);
		out[1] = v1.sub(v2).columnSums().transpose()
				.mul(alpha / option.batchsize);
		out[2] = h1.sub(h2).columnSums().transpose()
				.mul(alpha / option.batchsize);
		out[3] = MatrixFunctions.pow(v1.sub(v2), 2).rowSums().columnSums()
				.div(option.batchsize);
		return out;

	}
}
