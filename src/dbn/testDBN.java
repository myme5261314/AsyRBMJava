/**
 * 
 */
package dbn;

import java.util.Arrays;

import nn.NN;

import org.jblas.DoubleMatrix;

import util.MNISTImageReader;
import util.MNISTLabelReader;
import util.Option;

/**
 * @author peng
 * 
 */
public class testDBN {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		long begintime = System.currentTimeMillis();
		// {train_x, train_y, test_x, test_y}
		DoubleMatrix[] xy = new DoubleMatrix[4];
		getXY(xy);
		long endtime = System.currentTimeMillis();
		dispTime("Finish Load Data. ", begintime, endtime);
		Option op = new Option();
		op.numepochs = 30;
		op.momentum = 0;
		op.alpha = 1;
		op.push = 10;
		op.fetch = 20;
		int[] size = { 100, 100 };
		int repeat = 10;
		int thread = 20;
		double nRBM_time = 0;
		double[] aRBM_time = new double[thread-1];
		for (int i = 0; i < repeat; i++) {
			DBN dbn = new DBN(xy[0], op, size);
			dbn.setup();
			begintime = System.currentTimeMillis();
			dbn.train();
			endtime = System.currentTimeMillis();
			dispTime("Finish normal RBM. ", begintime, endtime);
			nRBM_time += (endtime-begintime)/1000/repeat;
		}

//		begintime = System.currentTimeMillis();
//		NN nn = new NN(dbn, 10);
//		nn.train(xy[0], xy[1], op);
//		double err = nn.test(xy[0], xy[1]);
//		System.out.println("The accuracy on train set is " + err);
//		err = nn.test(xy[2], xy[3]);
//		System.out.println("The accuracy on test set is " + err);
//		endtime = System.currentTimeMillis();
//		dispTime("Finish normalRBM's NN. ", begintime, endtime);
		
		for (int i = 1; i < thread; i++) {
			aRBM_time[i-1] = 0;
			for (int j = 0; j < repeat; j++) {
				AsyDBN asy_dbn = new AsyDBN(xy[0], op, size, i+1);
				asy_dbn.setup();
				begintime = System.currentTimeMillis();
				asy_dbn.train();
				endtime = System.currentTimeMillis();
				dispTime("Finish Asynchronous RBM. ", begintime, endtime);
				aRBM_time[i-1] += (endtime-begintime)/1000/repeat;
			}
		}
		System.out.println(nRBM_time);
		System.out.println(Arrays.toString(aRBM_time));
//		begintime = System.currentTimeMillis();
//		NN nn1 = new NN(asy_dbn, 10);
//		nn1.train(xy[0], xy[1], op);
//		double err1 = nn1.test(xy[0], xy[1]);
//		System.out.println("The accuracy on train set is " + err1);
//		err1 = nn1.test(xy[2], xy[3]);
//		System.out.println("The accuracy on test set is " + err1);
//		endtime = System.currentTimeMillis();
//		dispTime("Finish AsyRBM's NN. ", begintime, endtime);

	}

	public static void getXY(DoubleMatrix[] xy) {
		// xy [train_x, train_y, test_x, test_y]
		MNISTImageReader r = new MNISTImageReader("train-images-idx3-ubyte");
		xy[0] = r.getDataMat();
		MNISTLabelReader ry = new MNISTLabelReader("train-labels-idx1-ubyte");
		DoubleMatrix labels = ry.getDataMat();
		xy[1] = DoubleMatrix.zeros(labels.rows, 10);
		for (int i = 0; i < labels.length; i++) {
			xy[1].put(i, (int) labels.get(i), 1);
		}
		xy[0].divi(255);

		MNISTImageReader testimage = new MNISTImageReader(
				"t10k-images-idx3-ubyte");
		MNISTLabelReader testlabel = new MNISTLabelReader(
				"t10k-labels-idx1-ubyte");
		xy[2] = testimage.getDataMat();
		DoubleMatrix test_label = testlabel.getDataMat();
		xy[3] = DoubleMatrix.zeros(test_label.rows, 10);
		for (int i = 0; i < test_label.length; i++) {
			xy[3].put(i, (int) test_label.get(i), 1);
		}
		xy[2].divi(255);
	}

	public static void dispTime(long begin, long end) {
		dispTime("", begin, end);
	}

	public static void dispTime(String str, long begin, long end) {
		System.out.println(str + "Elapsed Time is:" + (double) (end - begin)
				/ 1000);
	}

}
