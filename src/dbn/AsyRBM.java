/**
 * 
 */
package dbn;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import util.Option;

/**
 * @author peng
 * 
 */
public class AsyRBM extends RBM {

	/**
	 * @param sz
	 * @param alpha
	 * @param momentum
	 */
	int thread_num = 4;

	public AsyRBM(int[] sz, double alpha, double momentum) {
		super(sz, alpha, momentum);
		// TODO Auto-generated constructor stub
	}

	public void train(DoubleMatrix x, Option option) {
		int m = x.rows;
		DoubleMatrix[] split_x = new DoubleMatrix[thread_num];
		int split_size = m / thread_num;
		for (int i = 0; i < split_x.length; i++) {
			split_x[i] = x.getRows(new IntervalRange(i * split_size, (i + 1)
					* split_size));
		}
		for (int i = 0; i < option.numepochs; i++) {
			Thread[] thread_list = new Thread[thread_num];
			for (int j = 0; j < thread_list.length; j++) {
				thread_list[j] = new Thread(new AsyWorkerRBM(split_x[j], this,
						option, size, alpha, momentum), "Thread" + j);
				thread_list[j].start();
			}
			for (int j = 0; j < thread_list.length; j++) {
				try {
					thread_list[j].join();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
