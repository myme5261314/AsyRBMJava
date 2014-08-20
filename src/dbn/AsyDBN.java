/**
 * 
 */
package dbn;

import org.jblas.DoubleMatrix;

import util.Option;

/**
 * @author peng
 *
 */
public class AsyDBN extends DBN {

	/**
	 * @param x
	 * @param op
	 * @param sz
	 */
	int thread_num = 2;
	public AsyDBN(DoubleMatrix x, Option op, int[] sz) {
		super(x, op, sz);
		// TODO Auto-generated constructor stub
	}
	
	public boolean setup() {
		rbm = new AsyRBM[this.size.length-1];
		boolean flag = true;
		for (int i = 0; i < this.size.length-1; i++) {
			int[] sz = {size[i], size[i+1]};
			rbm[i] = new AsyRBM( sz, option.alpha, option.momentum );
			flag &= rbm[i].setup();
		}
		return flag;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
