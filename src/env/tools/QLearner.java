package tools;

import java.util.*;
import java.util.logging.*;
import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab; // the lab environment that will be learnt 
  private int stateCount; // the number of possible states in the lab environment
  private int actionCount; // the number of possible actions in the lab environment
  private HashMap<Integer, double[][]> qTables; // a map for storing the qTables computed for different goals

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n="+ stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m="+ actionCount);

    qTables = new HashMap<>();
  }

  /**
    * Computes a Q matrix for the state space and action space of the lab, and against
    * a goal description. For example, the goal description can be of the form [z1level, z2Level],
    * where z1Level is the desired value of the light level in Zone 1 of the lab,
    * and z2Level is the desired value of the light level in Zone 2 of the lab.
    * For exercise 11, the possible goal descriptions are:
    * [0,0], [0,1], [0,2], [0,3], 
    * [1,0], [1,1], [1,2], [1,3], 
    * [2,0], [2,1], [2,2], [2,3], 
    * [3,0], [3,1], [3,2], [3,3].
    *
    *<p>
    * HINT: Use the methods of {@link LearningEnvironment} (implemented in {@link Lab})
    * to interact with the learning environment (here, the lab), e.g., to retrieve the
    * applicable actions, perform an action at the lab during learning etc.
    *</p>
    * @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
    * @param  episodesObj the number of episodes used for calculating the Q matrix
    * @param  alphaObj the learning rate with range [0,1].
    * @param  gammaObj the discount factor [0,1]
    * @param epsilonObj the exploration probability [0,1]
    * @param rewardObj the reward assigned when reaching the goal state
  **/
  @OPERATION
  public void calculateQ(Object[] goalDescription, Object episodesObj, Object alphaObj, Object gammaObj, Object epsilonObj, Object rewardObj) {
      
      // ensure that the right datatypes are used
      Integer episodes = Integer.valueOf(episodesObj.toString());
      Double alpha = Double.valueOf(alphaObj.toString());
      Double gamma = Double.valueOf(gammaObj.toString());
      Double epsilon = Double.valueOf(epsilonObj.toString());
      Integer reward = Integer.valueOf(rewardObj.toString());
      
      // Create a unique key for this goal
      String goalKey = Arrays.toString(goalDescription);
      
      // Initialize Q-table
      double[][] qTable = initializeQTable();
      
      // Get compatible goal states (states where the goal is achieved)
      List<Object> goalList = Arrays.asList(goalDescription);
      List<Integer> goalStates = lab.getCompatibleStates(goalList);
      
      LOGGER.info("Starting Q-learning for goal " + goalKey + " with " + episodes + " episodes");
      LOGGER.info("Goal states: " + goalStates);
      
      Random random = new Random();
      
      // Q-learning algorithm
      for (int episode = 0; episode < episodes; episode++) {
          // Randomize initial state by performing random actions
          for (int i = 0; i < 10; i++) {
              List<Integer> randomActions = lab.getApplicableActions(lab.readCurrentState());
              if (!randomActions.isEmpty()) {
                  int randomAction = randomActions.get(random.nextInt(randomActions.size()));
                  lab.performAction(randomAction);
              }
          }
          
          // Read initial state
          int currentState = lab.readCurrentState();
          
          // Episode loop
          int steps = 0;
          int maxSteps = 100; // Prevent infinite loops
          
          while (!goalStates.contains(currentState) && steps < maxSteps) {
              // Get applicable actions for current state
              List<Integer> applicableActions = lab.getApplicableActions(currentState);
              
              if (applicableActions.isEmpty()) {
                  break; // No actions available
              }
              
              // Epsilon-greedy action selection
              int selectedAction;
              if (random.nextDouble() < epsilon) {
                  // Exploration: random action
                  selectedAction = applicableActions.get(random.nextInt(applicableActions.size()));
              } else {
                  // Exploitation: best action based on Q-values
                  selectedAction = applicableActions.get(0);
                  double maxQ = qTable[currentState][selectedAction];
                  
                  for (int action : applicableActions) {
                      if (qTable[currentState][action] > maxQ) {
                          maxQ = qTable[currentState][action];
                          selectedAction = action;
                      }
                  }
              }
              
              // Perform action
              lab.performAction(selectedAction);
              
              // Observe new state
              int nextState = lab.readCurrentState();
              
              // Calculate reward
              double immediateReward = 0;
              if (goalStates.contains(nextState)) {
                  immediateReward = reward; // Goal achieved
              }
              
              // Find max Q-value for next state
              List<Integer> nextActions = lab.getApplicableActions(nextState);
              double maxNextQ = 0;
              if (!nextActions.isEmpty()) {
                  maxNextQ = qTable[nextState][nextActions.get(0)];
                  for (int action : nextActions) {
                      if (qTable[nextState][action] > maxNextQ) {
                          maxNextQ = qTable[nextState][action];
                      }
                  }
              }
              
              // Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
              double oldQ = qTable[currentState][selectedAction];
              double newQ = oldQ + alpha * (immediateReward + gamma * maxNextQ - oldQ);
              qTable[currentState][selectedAction] = newQ;
              
              // Move to next state
              currentState = nextState;
              steps++;
          }
          
          // Log progress every 100 episodes
          if ((episode + 1) % 100 == 0) {
              LOGGER.info("Completed episode " + (episode + 1) + "/" + episodes);
          }
      }
      
      // Store the Q-table for this goal
      qTables.put(goalKey.hashCode(), qTable);
      
      // Display Q-learning summary
      LOGGER.info("========== Q-LEARNING SUMMARY ==========");
      LOGGER.info("Goal: " + goalKey);
      LOGGER.info("Episodes: " + episodes);
      LOGGER.info("Parameters: alpha=" + alpha + ", gamma=" + gamma + ", epsilon=" + epsilon + ", reward=" + reward);
      LOGGER.info("Goal states found: " + goalStates.size() + " states");
      
      // Calculate statistics about the Q-table
      int nonZeroEntries = 0;
      double maxQValue = Double.NEGATIVE_INFINITY;
      double minQValue = Double.POSITIVE_INFINITY;
      int statesWithActions = 0;
      
      for (int s = 0; s < stateCount; s++) {
          boolean hasAction = false;
          for (int a = 0; a < actionCount; a++) {
              if (qTable[s][a] != 0.0) {
                  nonZeroEntries++;
                  hasAction = true;
                  maxQValue = Math.max(maxQValue, qTable[s][a]);
                  minQValue = Math.min(minQValue, qTable[s][a]);
              }
          }
          if (hasAction) statesWithActions++;
      }
      
      LOGGER.info("Q-table statistics:");
      LOGGER.info("  - States with learned actions: " + statesWithActions + "/" + stateCount);
      LOGGER.info("  - Non-zero Q-values: " + nonZeroEntries + "/" + (stateCount * actionCount));
      LOGGER.info("  - Max Q-value: " + String.format("%.2f", maxQValue));
      LOGGER.info("  - Min Q-value: " + String.format("%.2f", minQValue));
      
      // Show best actions for goal states
      LOGGER.info("Best actions for goal states:");
      for (int goalState : goalStates) {
          List<Integer> actions = lab.getApplicableActions(goalState);
          if (!actions.isEmpty()) {
              int bestAction = actions.get(0);
              double bestQ = qTable[goalState][bestAction];
              for (int action : actions) {
                  if (qTable[goalState][action] > bestQ) {
                      bestQ = qTable[goalState][action];
                      bestAction = action;
                  }
              }
              if (bestQ > 0) {
                  LOGGER.info("  - State " + goalState + ": Action " + bestAction + 
                            " (Q=" + String.format("%.2f", bestQ) + ")");
              }
          }
      }
      
      LOGGER.info("========================================");
      
      // Optionally print the full Q-table for small runs
      if (episodes <= 100) { // Only print for very small runs
          printQTable(qTable);
      }
  }

/**
* Returns information about the next best action based on a provided state and the QTable for
* a goal description. The returned information can be used by agents to invoke an action 
* using a ThingArtifact.
*
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  currentStateDescription the current state e.g. [2,2,true,false,true,true,2]
* @param  nextBestActionTag the (returned) semantic annotation of the next best action, e.g. "http://example.org/was#SetZ1Light"
* @param  nextBestActionPayloadTags the (returned) semantic annotations of the payload of the next best action, e.g. [Z1Light]
* @param nextBestActionPayload the (returned) payload of the next best action, e.g. true
**/
  @OPERATION
  public void getActionFromState(Object[] goalDescription, Object[] currentStateDescription,
      OpFeedbackParam<String> nextBestActionTag, OpFeedbackParam<Object[]> nextBestActionPayloadTags,
      OpFeedbackParam<Object[]> nextBestActionPayload) {
         
        // remove the following upon implementing Task 2.3!

        // sets the semantic annotation of the next best action to be returned 
        nextBestActionTag.set("http://example.org/was#SetZ1Light");

        // sets the semantic annotation of the payload of the next best action to be returned 
        Object payloadTags[] = { "Z1Light" };
        nextBestActionPayloadTags.set(payloadTags);

        // sets the payload of the next best action to be returned 
        Object payload[] = { true };
        nextBestActionPayload.set(payload);
      }

    /**
    * Print the Q matrix
    *
    * @param qTable the Q matrix
    */
  void printQTable(double[][] qTable) {
    System.out.println("Q matrix");
    for (int i = 0; i < qTable.length; i++) {
      System.out.print("From state " + i + ":  ");
     for (int j = 0; j < qTable[i].length; j++) {
      System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
  }

  /**
  * Initialize a Q matrix
  *
  * @return the Q matrix
  */
 private double[][] initializeQTable() {
    double[][] qTable = new double[this.stateCount][this.actionCount];
    for (int i = 0; i < stateCount; i++){
      for(int j = 0; j < actionCount; j++){
        qTable[i][j] = 0.0;
      }
    }
    return qTable;
  }
}
