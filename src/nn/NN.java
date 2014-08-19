/**
 * 
 */
package nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.util.Permutations;

import util.Option;
import util.Util;
import dbn.DBN;

/**
 * class NN for training the Neural Network.
 * 
 * @author peng
 * 
 * @param size
 *            the array of every layer neuron number.
 * @param layNum
 *            the number of layer.
 * @param activation_function
 *            Activation functions of hidden layers: 'sigm' (sigmoid) or
 *            'tanh_opt' (optimal tanh).
 * @param learningRate
 *            learning rate Note: typically needs to be lower when using 'sigm'
 *            activation function and non-normalized inputs.
 * @param momentum
 *            Momentum.
 * @param scaling_learningRate
 *            Scaling factor for the learning rate (each epoch)
 * @param weightPenaltyL2
 *            L2 regularization.
 * @param nonSparsityPenalty
 *            Non sparsity penalty.
 * @param sparsityTarget
 *            Sparsity target
 * @param inputZeroMaskedFraction
 *            Used for Denoising AutoEncoders
 * @param dropoutFraction
 *            Dropout level <a href =
 *            "http://www.cs.toronto.edu/~hinton/absps/dropout.pdf">http
 *            ://www.cs.toronto.edu/~hinton/absps/dropout.pdf</a>
 * @param testing
 *            Internal variable. nntest sets this to one.
 * @param output
 *            output unit 'sigm' (=logistic), 'softmax' and 'linear'
 * 
 */
public class NN {

	int[] size;
	int layerNum;
	String activation_function = "tanh_opt";
	double learningRate = 2;
	double momentum = 0.5;
	double scaling_learningRate = 1;
	double weightPenaltyL2 = 0;
	double nonSparsityPenalty = 0;
	double sparsityTarget = 0.05;
	double inputZeroMaskedFraction = 0;
	double dropoutFraction = 0;
	boolean testing = false;
	String output = "";

	DoubleMatrix[] W;
	DoubleMatrix[] vW;
	// average activations (for use with sparsity)
	private DoubleMatrix[] p;
	// Unit Activation
	private DoubleMatrix[] a;
	// Batch error and cost
	private DoubleMatrix e;
	private double cost = 0;
	// Drop Out
	private DoubleMatrix[] dropOutMask;
	// Derivative of W
	private DoubleMatrix[] dW;

	public NN(int[] sz) {
		this(sz, "sigm", "sigm");
	}

	public NN(int[] sz, String activation, String output) {
		size = sz;
		layerNum = size.length;
		this.activation_function = activation;
		this.output = output;
	}

	public boolean setup() {
		W = new DoubleMatrix[layerNum - 1];
		vW = new DoubleMatrix[layerNum - 1];
		p = new DoubleMatrix[layerNum];
		for (int i = 1; i < size.length; i++) {
			// weights and weight momentum
			W[i - 1] = DoubleMatrix.rand(size[i], size[i - 1] + 1).subi(0.5)
					.muli(2).muli(4)
					.muli(Math.sqrt(6 / (size[i] + size[i - 1])));
			vW[i - 1] = DoubleMatrix.zeros(W[i - 1].rows, W[i - 1].columns);
			// average activations (for use with sparsity)
			p[i] = DoubleMatrix.zeros(1, size[i]);
		}
		return true;
	}

	public NN(DBN dbn, int outputsize) {
		size = new int[dbn.size.length + 1];
		System.arraycopy(dbn.size, 0, size, 0, dbn.size.length);
		size[dbn.size.length] = outputsize;
		layerNum = size.length;
		this.activation_function = "sigm";
		this.output = "sigm";
		setup();
		for (int i = 0; i < dbn.rbm.length; i++) {
			W[i] = DoubleMatrix.concatHorizontally(dbn.rbm[i].c, dbn.rbm[i].W);
		}
	}

	public boolean train(DoubleMatrix x, DoubleMatrix y, Option option) {
		int m = x.rows;
		int batchsize = option.batchsize;
		int numepochs = option.numepochs;
		int numbatches = m / batchsize;

		initializeInternal(batchsize);
		DoubleMatrix error_list = new DoubleMatrix(numepochs, numbatches);

		for (int i = 0; i < numepochs; i++) {
			int[] index = Permutations.randomPermutation(m);
			for (int j = 0; j < numbatches; j++) {
				int[] oneIndex = new int[option.batchsize];
				System.arraycopy(index, j * option.batchsize, oneIndex, 0,
						option.batchsize);
				DoubleMatrix batch_x = x.getRows(oneIndex);
				// Add noise to input (for use in denoising autoencoder)
				if (inputZeroMaskedFraction != 0) {
					batch_x.muli(DoubleMatrix.rand(batch_x.rows,
							batch_x.columns).gt(inputZeroMaskedFraction));
				}
				DoubleMatrix batch_y = y.getRows(oneIndex);

				trainBatch(batch_x, batch_y);
				error_list.put(i, j, cost);
			}
			// Evaluation per epoch.
			eval(x, y);
			String str_perf = "; Full-batch train err = " + cost;
			String str_disp = "epoch " + (i + 1) + "/" + numepochs
					+ ". Mini-batch mean squared error on training set is "
					+ error_list.getRow(i).mean() + str_perf;
			System.out.println(str_disp);

			learningRate *= scaling_learningRate;

		}
		finalizeInternal();
		return true;
	}

	public double test(DoubleMatrix test_x, DoubleMatrix test_y) {
		DoubleMatrix labels = predict(test_x);
		double[] temp = new double[test_x.rows];
		int[] label = test_y.rowArgmaxs();
		for (int i = 0; i < test_x.rows; i++) {
			temp[i] = label[i];
		}
		DoubleMatrix expected = new DoubleMatrix(temp.length, 1, temp);
		return labels.eqi(expected).sum() / test_x.rows;
	}

	/**
	 * @param test_x
	 * @return
	 */
	private DoubleMatrix predict(DoubleMatrix test_x) {
		// TODO Auto-generated method stub
		testing = true;
		finalizeInternal();
		initializeInternal(test_x.rows);
		feedforward(test_x, DoubleMatrix.zeros(test_x.rows, size[layerNum - 1]));
		testing = false;
		int[] labels = a[layerNum - 1].rowArgmaxs();
		double[] temp = new double[labels.length];
		for (int i = 0; i < labels.length; i++) {
			temp[i] = labels[i];
		}
		return new DoubleMatrix(labels.length, 1, temp);
	}

	/**
	 * @param x
	 * @param y
	 */
	private void eval(DoubleMatrix x, DoubleMatrix y) {
		// TODO Auto-generated method stub
		feedforward(x, y);
	}

	private void initializeInternal(int batchsize) {
		a = new DoubleMatrix[layerNum];
		// a[0] = DoubleMatrix.zeros(batchsize, size[0] + 1);
		// for (int i = 1; i < layerNum; i++) {
		// a[i] = DoubleMatrix.zeros(batchsize, size[i]);
		// }
		e = DoubleMatrix.zeros(batchsize, size[layerNum - 1]);
		cost = 0;
		if (dropoutFraction > 0) {
			dropOutMask = new DoubleMatrix[layerNum];
		}
		dW = new DoubleMatrix[layerNum - 1];
	}

	private void finalizeInternal() {
		a = null;
		e = null;
		dropOutMask = null;
		dW = null;
	}

	private void trainBatch(DoubleMatrix batch_x, DoubleMatrix batch_y) {
		feedforward(batch_x, batch_y);
		backpropagation();
		applaygrads();
	}

	private void feedforward(DoubleMatrix batch_x, DoubleMatrix batch_y) {
		int n = layerNum;
		int m = batch_x.rows;
		DoubleMatrix x = DoubleMatrix.concatHorizontally(
				DoubleMatrix.ones(m, 1), batch_x);
		a[0] = x;

		// feedforward pass
		for (int i = 1; i < n - 1; i++) {
			// Calculate the unit's outputs (including the bias term)
			a[i] = a[i - 1].mmul(W[i - 1].transpose());
			switch (activation_function) {
			case "sigm":
				a[i] = Util.sigmoidMat(a[i]);
				break;
			case "tanh_opt":
				a[i] = Util.tanh_optMat(a[i]);
			default:
				break;
			}
			// Drop out
			if (dropoutFraction > 0) {
				if (testing) {
					a[i].muli(1 - dropoutFraction);
				} else {
					dropOutMask[i] = DoubleMatrix.rand(a[i].rows, a[i].columns)
							.gti(dropoutFraction);
					a[i].muli(dropOutMask[i]);
				}
			}
			// calculate running exponential activations for use with sparsity
			if (nonSparsityPenalty > 0) {
				p[i].muli(0.99).addi(a[i].columnMeans().mul(0.01));
			}
			// Add the bias term
			a[i] = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(m, 1),
					a[i]);
		}

		// Output unit activation
		a[n - 1] = a[n - 2].mmul(W[n - 2].transpose());
		switch (output) {
		case "sigm":
			a[n - 1] = Util.sigmoidMat(a[n - 1]);
			break;
		case "linear":
			break;
		case "softmax":
			a[n - 1] = MatrixFunctions.exp(a[n - 1].sub(a[n - 1].rowMaxs()
					.repmat(1, a[n - 1].columns)));
			a[n - 1].rdivi(a[n - 1].rowSums().repmat(1, a[n - 1].columns));
			break;
		default:
			break;
		}
		e = batch_y.sub(a[n - 1]);
		switch (output) {
		case "sigm":
		case "linear":
			cost = MatrixFunctions.pow(e, 2).sum() / 2 / m;
			break;
		case "softmax":
			cost = batch_y.mul(MatrixFunctions.log(a[n - 1])).sum() * -1 / m;
			break;
		default:
			break;
		}

	}

	private void backpropagation() {
		int n = layerNum;
		DoubleMatrix sparsityError;
		DoubleMatrix[] d = new DoubleMatrix[n];

		switch (output) {
		case "sigm":
			d[n - 1] = e.mul(a[n - 1].mul(a[n - 1].sub(1)));
			break;
		case "linear":
		case "softmax":
			d[n - 1] = e.mul(-1);
		default:
			break;
		}

		for (int i = n - 2; i > 0; i--) {
			// Derivative of the activation function
			DoubleMatrix d_act;
			switch (activation_function) {
			case "sigm":
				d_act = a[i].mul(a[i].rsub(1));
				break;
			case "tanh_opt":
				double temp = 1.7159 * 2 / 3 * (1 - Math.pow(1.0d / 1.7159, 2));
				d_act = MatrixFunctions.pow(a[i], 2).mul(temp);
				break;
			default:
				d_act = DoubleMatrix.zeros(a[i].rows, a[i].columns);
				break;
			}

			if (nonSparsityPenalty > 0) {
				DoubleMatrix pi = p[i].repmat(a[i].rows, 1);
				sparsityError = DoubleMatrix.concatHorizontally(
						DoubleMatrix.zeros(a[i].rows, 1),
						pi.rdiv(-1 * sparsityTarget).add(
								pi.rsub(1).rdiv(1 - sparsityTarget))).mul(
						nonSparsityPenalty);
			} else {
				sparsityError = DoubleMatrix.zeros(a[i].rows, p[i].columns + 1);
			}

			// Backpropagate first derivatives
			if (i == n - 2) { // in this case in d{n} there is not the bias term
								// to be removed
				d[i] = d[i + 1].mmul(W[i]).add(sparsityError).mul(d_act);
			} else { // in this case in d{i} the bias term has to be removed
				int[] cindex = Util.generateIdx(1, d[i + 1].columns - 1);
				d[i] = d[i + 1].getColumns(cindex).mmul(W[i])
						.addi(sparsityError).muli(d_act);
			}

			if (dropoutFraction > 0) {
				d[i].muli(DoubleMatrix.concatHorizontally(
						DoubleMatrix.ones(d[i].rows, 1), dropOutMask[i]));
			}
		}

		for (int i = 0; i < n - 1; i++) {
			if (i == n - 2) {
				dW[i] = d[i + 1].transpose().mmul(a[i]).div(d[i + 1].rows);
			} else {
				int[] cindex = Util.generateIdx(1, d[i + 1].columns - 1);
				dW[i] = d[i + 1].getColumns(cindex).transpose().mmul(a[i])
						.divi(d[i + 1].rows);
			}
		}

	}

	private void applaygrads() {
		int n = layerNum;
		DoubleMatrix t_dW;
		for (int i = 0; i < n - 1; i++) {
			if (weightPenaltyL2 > 0) {
				int[] cindex = Util.generateIdx(1, W[i].columns - 1);
				t_dW = dW[i].add(DoubleMatrix.concatHorizontally(
						DoubleMatrix.zeros(W[i].rows, 1),
						W[i].getColumns(cindex)).muli(weightPenaltyL2));
			} else {
				t_dW = dW[i].dup();
			}
			t_dW.muli(learningRate);
			if (momentum > 0) {
				vW[i].muli(momentum).addi(t_dW);
				t_dW = vW[i].dup();
			}

			W[i].subi(t_dW);
		}
	}

}
