package tools;

import java.util.*;
import java.util.logging.*;

import javax.swing.Action;

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
  public void calculateQ(Object[] goalDescription , Object episodesObj, Object alphaObj, Object gammaObj, Object epsilonObj, Object rewardObj) {
    
    // ensure that the right datatypes are used
    Integer episodes = Integer.valueOf(episodesObj.toString());
    Double alpha = Double.valueOf(alphaObj.toString());
    Double gamma = Double.valueOf(gammaObj.toString());
    Double epsilon = Double.valueOf(epsilonObj.toString());
    Integer reward = Integer.valueOf(rewardObj.toString());

    LOGGER.info("Starting Q-Learning for goal: " + Arrays.toString(goalDescription));
    LOGGER.info("Parameters - Episodes: " + episodes + ", Alpha: " + alpha + ", Gamma: " + gamma + ", Epsilon: " + epsilon + ", Reward: " + reward);

    // Initialize Q-table
    double[][] qTable = initializeQTable();
    
    // Convert goal description to list for compatibility checking
    List<Object> goalList = Arrays.asList(goalDescription);
    
    // Get all states that are compatible with the goal (terminal states)
    List<Integer> terminalStates = lab.getCompatibleStates(goalList);
    LOGGER.info("Terminal states for goal " + Arrays.toString(goalDescription) + ": " + terminalStates);
    
    Random random = new Random();
    
    // Q-Learning episodes
    for (int episode = 0; episode < episodes; episode++) {
      
      // Randomize initial state by performing random actions
      randomizeEnvironment(random, 5); // Perform 5 random actions to randomize state
      
      int currentState = lab.readCurrentState();
      int steps = 0;
      int maxStepsPerEpisode = 50; // Prevent infinite loops
      
      // Run episode until terminal state or max steps
      while (!terminalStates.contains(currentState) && steps < maxStepsPerEpisode) {
        
        // Get applicable actions for current state
        List<Integer> applicableActions = lab.getApplicableActions(currentState);
        
        if (applicableActions.isEmpty()) {
          LOGGER.warning("No applicable actions in state " + currentState);
          break;
        }
        
        // Choose action using epsilon-greedy policy
        int action;
        if (random.nextDouble() < epsilon) {
          // Explore: choose random action
          action = applicableActions.get(random.nextInt(applicableActions.size()));
        } else {
          // Exploit: choose best action based on current Q-values
          action = getBestAction(qTable, currentState, applicableActions);
        }
        
        // Perform the action
        lab.performAction(action);
        
        // Read new state
        int newState = lab.readCurrentState();
        
        // Calculate reward
        double immediateReward = calculateReward(newState, terminalStates, reward, action);
        
        // Q-Learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        double oldQValue = qTable[currentState][action];
        double maxNextQValue = getMaxQValue(qTable, newState, lab.getApplicableActions(newState));
        double newQValue = oldQValue + alpha * (immediateReward + gamma * maxNextQValue - oldQValue);
        qTable[currentState][action] = newQValue;
        
        // Move to new state
        currentState = newState;
        steps++;
      }
      
      // Log progress every 100 episodes
      if ((episode + 1) % 100 == 0) {
        LOGGER.info("Completed episode " + (episode + 1) + "/" + episodes);
      }
    }
    
    // Store the computed Q-table using goal description hash as key
    int goalHash = Arrays.hashCode(goalDescription);
    qTables.put(goalHash, qTable);
    
    LOGGER.info("Q-Learning completed for goal: " + Arrays.toString(goalDescription));
    LOGGER.info("Q-table stored with hash key: " + goalHash);
    
    // Print Q-table for debugging (optional)
    printQTable(qTable);
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

  /**
   * Randomizes the environment state by performing random actions
   */
  private void randomizeEnvironment(Random random, int numActions) {
    for (int i = 0; i < numActions; i++) {
      int currentState = lab.readCurrentState();
      List<Integer> applicableActions = lab.getApplicableActions(currentState);
      if (!applicableActions.isEmpty()) {
        int randomAction = applicableActions.get(random.nextInt(applicableActions.size()));
        lab.performAction(randomAction);
      }
    }
  }

  /**
   * Gets the best action for a given state based on Q-values
   */
  private int getBestAction(double[][] qTable, int state, List<Integer> applicableActions) {
    int bestAction = applicableActions.get(0);
    double maxQValue = qTable[state][bestAction];
    
    for (int action : applicableActions) {
      if (qTable[state][action] > maxQValue) {
        maxQValue = qTable[state][action];
        bestAction = action;
      }
    }
    return bestAction;
  }

  /**
   * Gets the maximum Q-value for a given state
   */
  private double getMaxQValue(double[][] qTable, int state, List<Integer> applicableActions) {
    if (applicableActions.isEmpty()) {
      return 0.0;
    }
    
    double maxQValue = qTable[state][applicableActions.get(0)];
    for (int action : applicableActions) {
      if (qTable[state][action] > maxQValue) {
        maxQValue = qTable[state][action];
      }
    }
    return maxQValue;
  }

  /**
   * Calculates the reward for reaching a state
   */
  private double calculateReward(int state, List<Integer> terminalStates, int goalReward, int action) {
    double reward = 0.0;
    
    // Goal achievement reward
    if (terminalStates.contains(state)) {
      reward += goalReward;
    }
    
    // Energy cost penalties (based on action performed)
    Action actionObj = lab.getAction(action);
    if (actionObj != null) {
      String actionTag = actionObj.getActionTag();
      Object[] payload = actionObj.getPayload();
      
      // Light actions cost more energy
      if ((actionTag.contains("SetZ1Light") || actionTag.contains("SetZ2Light")) && 
          payload.length > 0 && Boolean.TRUE.equals(payload[0])) {
        reward -= 50; // High energy cost for turning on lights
      }
      
      // Blind actions cost less energy
      if ((actionTag.contains("SetZ1Blinds") || actionTag.contains("SetZ2Blinds")) && 
          payload.length > 0 && Boolean.TRUE.equals(payload[0])) {
        reward -= 1; // Low energy cost for raising blinds
      }
    }
    
    return reward;
  }
}
