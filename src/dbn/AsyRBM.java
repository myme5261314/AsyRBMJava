/**
 * 
 */
package dbn;

import org.jblas.DoubleMatrix;
import org.jblas.util.Permutations;

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
	int thread_num;
	double avg_error = 0;

	public AsyRBM(int[] sz, double alpha, double momentum, int thread_num) {
		super(sz, alpha, momentum);
		this.thread_num = thread_num;
		// TODO Auto-generated constructor stub
	}

	public void train(DoubleMatrix x, Option option) {
		int m = x.rows;
		DoubleMatrix[] split_x = new DoubleMatrix[thread_num];
		int split_size = m / thread_num;
		int[] index = Permutations.randomPermutation(m);
		for (int i = 0; i < split_x.length; i++) {
			int[] oneIndex = new int[split_size];
			System.arraycopy(index, i * split_size, oneIndex, 0, split_size);
			split_x[i] = x.getRows(oneIndex);
		}
		for (int i = 0; i < option.numepochs; i++) {
			avg_error = 0;
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
			System.out.println("Epoch " + i
					+ ". Average Reconstruction Error is: " + avg_error/thread_num);
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
