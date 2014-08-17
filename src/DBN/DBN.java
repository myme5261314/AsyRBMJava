package DBN;


import org.jblas.DoubleMatrix;

import util.Option;

public class DBN {
	public DoubleMatrix x;
	public Option option;
	public int[] size;
	public RBM[] rbm;
	
	public DBN(DoubleMatrix x, Option op, int[] sz) {
		this.x = x;
		assert(x!=null);
		assert(op!=null);
		assert(sz!=null);
		this.option = op;
		int[] n = {x.columns};
		this.size = new int[sz.length+1];
		System.arraycopy(n, 0, this.size, 0, 1);
		System.arraycopy(sz, 0, this.size, 1, sz.length);
	}
	
	public boolean setup() {
		rbm = new RBM[this.size.length-1];
		boolean flag = true;
		for (int i = 0; i < this.size.length-1; i++) {
			int[] sz = {size[i], size[i+1]};
			rbm[i] = new RBM( sz, option.alpha, option.momentum );
			flag &= rbm[i].setup();
		}
		return flag;
	}
	
	public void train() {
		rbm[0].train(x, option);
		DoubleMatrix data = x;
		for (int i = 1; i < this.size.length-1; i++) {
			data = rbm[i-1].rbmUp(data);
			rbm[i].train(data, option);
		}
	}
}
