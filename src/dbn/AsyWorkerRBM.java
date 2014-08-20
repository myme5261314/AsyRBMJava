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
	DoubleMatrix part_x;
	DoubleMatrix accrued_W, accrued_b, accrued_c;
	AsyRBM server_rbm;
	Option op;
	int n_push = 10;
	int n_fetch = 50;

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
		ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
		for (int i = 0; i < numbatches; i++) {
			if (step % n_fetch == 0) {
				lock.readLock().lock();
				fetchFromServer(server_rbm);
				lock.readLock().unlock();
			}

			int[] oneIndex = new int[op.batchsize];
			System.arraycopy(index, i * op.batchsize, oneIndex, 0, op.batchsize);
			DoubleMatrix batch = part_x.getRows(oneIndex);
			DoubleMatrix[] grad = computeGradient(batch, op);
			vW.muli(momentum).addi(grad[0]);
			vb.muli(momentum).addi(grad[1]);
			vc.muli(momentum).addi(grad[2]);
			accrued_W.addi(vW);
			accrued_b.addi(vb);
			accrued_c.addi(vc);
			W.addi(vW);
			b.addi(vb);
			c.addi(vc);
			error += grad[3].scalar();
			error_list.add(grad[3].scalar());

			if (step % n_push == 0) {
				lock.writeLock().lock();
				pushToServer(server_rbm);
				lock.writeLock().unlock();
			}
		}
		System.out.println(Thread.currentThread().getName()
				+ ". Average Reconstruction Error is: " + (error / numbatches));
		clearAccrued();

	}

	public boolean setup() {
		super.setup();
		accrued_W = DoubleMatrix.zeros(W.rows, W.columns);
		accrued_b = DoubleMatrix.zeros(b.rows, b.columns);
		accrued_c = DoubleMatrix.zeros(c.rows, c.columns);
		return true;
	}

	private void resetAccrued() {
		accrued_W.fill(0);
		accrued_b.fill(0);
		accrued_c.fill(0);
	}

	private void clearAccrued() {
		accrued_W = null;
		accrued_b = null;
		accrued_c = null;
	}

	/**
	 * @param server_rbm2
	 */
	private void pushToServer(AsyRBM rbm) {
		// TODO Auto-generated method stub
		rbm.W.addi(accrued_W);
		rbm.b.addi(accrued_b);
		rbm.c.addi(accrued_c);
		resetAccrued();
	}

	/**
	 * @param server_rbm2
	 */
	private void fetchFromServer(AsyRBM rbm) {
		// TODO Auto-generated method stub
		W.copy(rbm.W);
		b.copy(rbm.b);
		c.copy(rbm.c);
	}

}
