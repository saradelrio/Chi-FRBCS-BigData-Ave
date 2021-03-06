package org.apache.mahout.classifier.chi_rw.builder;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.chi_rw.data.Data;
import org.apache.mahout.classifier.chi_rw.data.Dataset;
import org.apache.mahout.classifier.chi_rw.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Fuzzy_ChiBuilder {
  
  private static final Logger log = LoggerFactory.getLogger(Fuzzy_ChiBuilder.class);	
  int nClasses, nLabels, combinationType, inferenceType, ruleWeight;
  DataBase dataBase;
  RuleBase ruleBase;
  
  public void setNLabels(int nLabels) {
    this.nLabels = nLabels;
  }

  public void setCombinationType(int combinationType) {
    this.combinationType = combinationType;
  }	
  
  public void setInferenceType(int inferenceType) {
    this.inferenceType = inferenceType;
  }	
  
  public void setRuleWeight(int ruleWeight) {
    this.ruleWeight = ruleWeight;
  }	
  
  public DataBase getDataBase() {
    return this.dataBase;
  }	
  
  public RuleBase getRuleBase() {
    return this.ruleBase;
  }	

  public void build(Data data, Context context) {
    //We do here the algorithm's operations

	Dataset dataset = data.getDataset();
	 
	nClasses = dataset.nblabels();
	
	//Gets the number of input attributes of the data-set
	int nInputs = dataset.nbAttributes() - 1;
	
	//It returns the class labels
	String clases[] = dataset.labels();
	
	dataBase = new DataBase(nInputs, nLabels, data.getRanges(), data.getNames());
	
	ruleBase = new RuleBase(dataBase, inferenceType, combinationType, ruleWeight, data.getNames(), clases);	
	
	ruleBase.Generation(data, context);
	
  }


}
