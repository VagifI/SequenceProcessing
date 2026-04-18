package SequenceProcessing.Functions;

import java.io.Serializable;
import java.util.ArrayList;

import ComputationalGraph.Function.Function;
import Math.Tensor;

public class AdditionByConstant implements Function, Serializable {

    private final double constant;

    public AdditionByConstant(double constant) {
        this.constant = constant;
    }

    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) tensor.getData();
        for (double val : tensorValues) {
            values.add(val + constant);
        }
        return new Tensor(values, tensor.getShape());
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor tensor1) {
        return tensor1;
    }
}
