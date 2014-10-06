/**
 * 
 */
package dbn;

import java.util.concurrent.locks.*;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;
import org.jblas.util.Permutations;

import util.Option;

/**
 * @author peng
 * 
 */
public class AsyWorkerRBM extends RBM implements Runnable {

	/**
	 * @param sz
	 * @param alpha
	 * @param momentum
	 */
	final static ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
//	static int enter_fetch = 0;
//	static int enter_push = 0;
	DoubleMatrix part_x;
	AsyRBM server_rbm;
	Option op;

	public AsyWorkerRBM(DoubleMatrix part_x, AsyRBM server_rbm, Option option,
			int[] sz, double alpha, double momentum) {
		super(sz, alpha, momentum);
		// TODO Auto-generated constructor stub
		this.part_x = part_x;
		this.server_rbm = server_rbm;
		op = option;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Runnable#run()
	 */
	@Override
	public void run() {
		// TODO Auto-generated method stub
		setup();
		int step = 0;
		double error = 0;
		ArrayList<Double> error_list = new ArrayList<Double>();
		int m = part_x.rows;
		int numbatches = m / op.batchsize;
		int[] index = Permutations.randomPermutation(m);

		for (int i = 0; i < numbatches; i++) {
			if (step % op.fetch == 0) {
				lock.readLock().lock();
//				System.out.println(Thread.currentThread().getName()
//						+ " enter the " + enter_fetch + "th fetch.");
				fetchFromServer(server_rbm);
//				System.out.println(Thread.currentThread().getName()
//						+ " exit the " + enter_fetch + "th fetch.");
//				enter_fetch++;
				lock.readLock().unlock();
			}

			int[] oneIndex = new int[op.batchsize];
			System.arraycopy(index, i * op.batchsize, oneIndex, 0, op.batchsize);
			DoubleMatrix batch = part_x.getRows(oneIndex);
			DoubleMatrix[] grad = computeGradient(batch, op);
			vW.muli(momentum).addi(grad[0]);
			vb.muli(momentum).addi(grad[1]);
			vc.muli(momentum).addi(grad[2]);
			W.addi(vW);
			b.addi(vb);
			c.addi(vc);
			error += grad[3].scalar();
			error_list.add(grad[3].scalar());

			if (step % op.push == 0) {
				lock.writeLock().lock();
//				System.out.println(Thread.currentThread().getName()
//						+ " enter the " + enter_push + "th push.");
				pushToServer(server_rbm);
//				System.out.println(Thread.currentThread().getName()
//						+ " exit the " + enter_push + "th push.");
//				enter_push++;
				lock.writeLock().unlock();
			}
			step++;
		}
//		System.out.println(Thread.currentThread().getName()
//				+ ". Average Reconstruction Error is: " + (error / numbatches));
		server_rbm.avg_error += error/numbatches;

	}

	public boolean setup() {
		super.setup();
		return true;
	}

	private void resetAccrued() {
		vW.fill(0);
		vb.fill(0);
		vc.fill(0);
	}


	/**
	 * @param server_rbm2
	 */
	private void pushToServer(AsyRBM rbm) {
		// TODO Auto-generated method stub
		double w2 = 1.0d/rbm.thread_num;
		double w1 = 1-w2;
		rbm.W.muli(w1).addi(W.mul(w2));
		rbm.b.muli(w1).addi(b.mul(w2));
		rbm.c.muli(w1).addi(c.mul(w2));
		resetAccrued();
	}

	/**
	 * @param server_rbm2
	 */
	private void fetchFromServer(AsyRBM rbm) {
		// TODO Auto-generated method stub
		W.copy(rbm.W);
		b.copy(rbm.b);
//		assert(b.ne(rbm.b).sum()==0);
		c.copy(rbm.c);
//		vW.fill(0);
//		vb.fill(0);
//		vc.fill(0);
	}

}
