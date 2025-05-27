package tools;

import java.util.*;
import java.util.logging.*;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
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
      
      if (goalStates.isEmpty()) {
          LOGGER.severe("No goal states found for " + goalKey + "! Q-learning cannot proceed.");
          return;
      }
      
      Random random = new Random();
      int goalAchievements = 0;
      
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
              
              // Calculate reward with energy costs
              double immediateReward = calculateReward(nextState, goalStates, reward, selectedAction);
              
              // Track goal achievements
              if (goalStates.contains(nextState)) {
                  goalAchievements++;
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
      
      LOGGER.info("Q-learning completed for goal " + goalKey);
      LOGGER.info("Goal achievements: " + goalAchievements + "/" + episodes + " (" + 
                  String.format("%.1f%%", 100.0 * goalAchievements / episodes) + ")");
      
      // Log Q-table statistics
      logQTableStatistics(qTable, goalDescription);
      
      // Optionally print the Q-table for small state spaces
      if (this.stateCount <= 50) {
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
         
    try {
      // Get the Q-table for this goal
      String goalKey = Arrays.toString(goalDescription);
      double[][] qTable = qTables.get(goalKey.hashCode());
      
      if (qTable == null) {
        LOGGER.severe("No Q-table found for goal: " + goalKey);
        setDefaultAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
      }
      
      // Convert current state description to state index
      int currentStateIndex = convertStateDescriptionToIndex(currentStateDescription);
      
      if (currentStateIndex == -1) {
        LOGGER.severe("Could not convert state to valid index: " + Arrays.toString(currentStateDescription));
        setDefaultAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
      }
      
      // Get applicable actions for current state
      List<Integer> applicableActions = lab.getApplicableActions(currentStateIndex);
      
      if (applicableActions.isEmpty()) {
        LOGGER.warning("No applicable actions for state: " + currentStateIndex);
        setDefaultAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
      }
      
      // Find the action with highest Q-value
      int bestAction = applicableActions.get(0);
      double maxQValue = qTable[currentStateIndex][bestAction];
      
      for (int action : applicableActions) {
        double qValue = qTable[currentStateIndex][action];
        if (qValue > maxQValue) {
          maxQValue = qValue;
          bestAction = action;
        }
      }
      
      // Get the action object
      Action actionObj = lab.getAction(bestAction);
      
      if (actionObj == null) {
        LOGGER.severe("Could not retrieve action object for action index: " + bestAction);
        setDefaultAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
      }
      
      // Set the return values
      nextBestActionTag.set(actionObj.getActionTag());
      nextBestActionPayloadTags.set(actionObj.getPayloadTags());
      nextBestActionPayload.set(actionObj.getPayload());
      
      LOGGER.info("Best action for goal " + goalKey + " from state index " + currentStateIndex + 
                  ": " + actionObj.getActionTag() + " with Q-value: " + maxQValue);
                  
    } catch (Exception e) {
      LOGGER.severe("Error in getActionFromState: " + e.getMessage());
      e.printStackTrace();
      setDefaultAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
    }
  }

  /**
   * Operation to read the current state of the lab environment
   */
  @OPERATION
  public void readCurrentLabState(OpFeedbackParam<Object[]> currentStateArray) {
    try {
      int currentStateIndex = lab.readCurrentState();
      List<List<Integer>> stateList = new ArrayList<>(lab.stateSpace);
      
      if (currentStateIndex >= 0 && currentStateIndex < stateList.size()) {
        List<Integer> stateVector = stateList.get(currentStateIndex);
        Object[] stateArray = new Object[stateVector.size()];
        for (int i = 0; i < stateVector.size(); i++) {
          stateArray[i] = stateVector.get(i);
        }
        currentStateArray.set(stateArray);
        LOGGER.info("Read current lab state: " + Arrays.toString(stateArray));
      } else {
        LOGGER.warning("Invalid state index: " + currentStateIndex);
        Object[] defaultState = {0, 0, 0, 0, 0, 0, 1};
        currentStateArray.set(defaultState);
      }
    } catch (Exception e) {
      LOGGER.severe("Error reading current lab state: " + e.getMessage());
      Object[] fallbackState = {1, 1, 0, 0, 0, 0, 2};
      currentStateArray.set(fallbackState);
    }
  }

  /**
   * Operation to get a formatted state description
   */
  @OPERATION  
  public void getStateDescription(OpFeedbackParam<String> stateDescription) {
    try {
      int currentStateIndex = lab.readCurrentState();
      List<List<Integer>> stateList = new ArrayList<>(lab.stateSpace);
      
      if (currentStateIndex >= 0 && currentStateIndex < stateList.size()) {
        List<Integer> state = stateList.get(currentStateIndex);
        String description = String.format(
          "State[%d]: Z1Level=%d, Z2Level=%d, Z1Light=%s, Z2Light=%s, Z1Blinds=%s, Z2Blinds=%s, Sunshine=%d",
          currentStateIndex,
          state.get(0), state.get(1),
          state.get(2) == 1 ? "ON" : "OFF",
          state.get(3) == 1 ? "ON" : "OFF", 
          state.get(4) == 1 ? "UP" : "DOWN",
          state.get(5) == 1 ? "UP" : "DOWN",
          state.get(6)
        );
        stateDescription.set(description);
        LOGGER.info("Current state description: " + description);
      } else {
        stateDescription.set("Invalid state index: " + currentStateIndex);
      }
    } catch (Exception e) {
      stateDescription.set("Error reading state: " + e.getMessage());
    }
  }

  /**
   * Export all computed Q-tables to files
   */
  @OPERATION
  public void exportAllQTables() {
    LOGGER.info("Exporting all Q-tables...");
    
    for (Map.Entry<Integer, double[][]> entry : qTables.entrySet()) {
      int goalHash = entry.getKey();
      double[][] qTable = entry.getValue();
      
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      String filename = "qtable_goal_" + goalHash + "_" + timestamp + ".csv";
      
      exportQTableToCSV(qTable, filename, goalHash);
      exportQTableToText(qTable, filename.replace(".csv", ".txt"), goalHash);
    }
    
    exportQTableSummary();
    LOGGER.info("Q-table export completed!");
  }

  /**
   * Export a specific Q-table by goal description
   */
  @OPERATION
  public void exportQTableForGoal(Object[] goalDescription, String filename) {
    String goalKey = Arrays.toString(goalDescription);
    double[][] qTable = qTables.get(goalKey.hashCode());
    
    if (qTable != null) {
      exportQTableToCSV(qTable, filename + ".csv", goalKey.hashCode());
      exportQTableToText(qTable, filename + ".txt", goalKey.hashCode());
      LOGGER.info("Exported Q-table for goal " + goalKey + " to " + filename);
    } else {
      LOGGER.warning("No Q-table found for goal: " + goalKey);
    }
  }

  /**
   * Print Q-table in a formatted way for reports
   */
  @OPERATION
  public void printQTableForReport(Object[] goalDescription) {
    String goalKey = Arrays.toString(goalDescription);
    double[][] qTable = qTables.get(goalKey.hashCode());
    
    if (qTable != null) {
      System.out.println("\n" + "=".repeat(80));
      System.out.println("Q-TABLE FOR GOAL: " + goalKey);
      System.out.println("Goal Hash: " + goalKey.hashCode());
      System.out.println("State Space Size: " + qTable.length);
      System.out.println("Action Space Size: " + qTable[0].length);
      System.out.println("=".repeat(80));
      
      System.out.print("State\\Action");
      for (int a = 0; a < qTable[0].length; a++) {
        System.out.printf("%12s", "A" + a);
      }
      System.out.println();
      
      for (int s = 0; s < Math.min(20, qTable.length); s++) { // Show first 20 states
        System.out.printf("%-12s", "S" + s);
        for (int a = 0; a < qTable[s].length; a++) {
          System.out.printf("%12.3f", qTable[s][a]);
        }
        System.out.println();
      }
      
      if (qTable.length > 20) {
        System.out.println("... (showing first 20 states of " + qTable.length + " total states)");
      }
      
      System.out.println("=".repeat(80) + "\n");
    } else {
      System.out.println("No Q-table found for goal: " + goalKey);
    }
  }

  /**
   * Get Q-table statistics
   */
  @OPERATION
  public void getQTableStats(Object[] goalDescription, OpFeedbackParam<String> stats) {
    String goalKey = Arrays.toString(goalDescription);
    double[][] qTable = qTables.get(goalKey.hashCode());
    
    if (qTable != null) {
      double max = Double.NEGATIVE_INFINITY;
      double min = Double.POSITIVE_INFINITY;
      double sum = 0.0;
      int nonZeroCount = 0;
      int totalEntries = qTable.length * qTable[0].length;
      
      for (int i = 0; i < qTable.length; i++) {
        for (int j = 0; j < qTable[i].length; j++) {
          double value = qTable[i][j];
          max = Math.max(max, value);
          min = Math.min(min, value);
          sum += value;
          if (Math.abs(value) > 0.001) nonZeroCount++;
        }
      }
      
      double avg = sum / totalEntries;
      double sparsity = 1.0 - ((double) nonZeroCount / totalEntries);
      
      String statistics = String.format(
        "Q-Table Statistics for Goal %s:\n" +
        "  Size: %d states × %d actions = %d entries\n" +
        "  Max Q-value: %.6f\n" +
        "  Min Q-value: %.6f\n" +
        "  Average Q-value: %.6f\n" +
        "  Non-zero entries: %d (%.1f%%)\n" +
        "  Sparsity: %.1f%%",
        goalKey,
        qTable.length, qTable[0].length, totalEntries,
        max, min, avg,
        nonZeroCount, (100.0 * nonZeroCount / totalEntries),
        (100.0 * sparsity)
      );
      
      stats.set(statistics);
      System.out.println(statistics);
    } else {
      stats.set("No Q-table found for goal: " + goalKey);
    }
  }

  // ==================== PRIVATE HELPER METHODS ====================

  /**
   * Calculate reward including energy costs
   */
  private double calculateReward(int state, List<Integer> goalStates, int goalReward, int action) {
    double reward = 0.0;
    
    // Goal achievement reward
    if (goalStates.contains(state)) {
      reward += goalReward;
    }
    
    // Energy cost penalties
    Action actionObj = lab.getAction(action);
    if (actionObj != null) {
      String actionTag = actionObj.getActionTag();
      Object[] payload = actionObj.getPayload();
      
      // Light actions cost more energy
      if ((actionTag.contains("SetZ1Light") || actionTag.contains("SetZ2Light")) && 
          payload.length > 0 && Boolean.TRUE.equals(payload[0])) {
        reward -= 50;
      }
      
      // Blind actions cost less energy
      if ((actionTag.contains("SetZ1Blinds") || actionTag.contains("SetZ2Blinds")) && 
          payload.length > 0 && Boolean.TRUE.equals(payload[0])) {
        reward -= 1;
      }
    }
    
    return reward;
  }

  /**
   * Convert state description to state index
   */
  private int convertStateDescriptionToIndex(Object[] stateDescription) {
    try {
      List<Integer> discretizedState = new ArrayList<>();
      
      if (stateDescription.length >= 7) {
        for (int i = 0; i < 7 && i < stateDescription.length; i++) {
          if (stateDescription[i] instanceof Boolean) {
            discretizedState.add(((Boolean) stateDescription[i]) ? 1 : 0);
          } else {
            discretizedState.add(((Number) stateDescription[i]).intValue());
          }
        }
      } else {
        LOGGER.severe("State description has insufficient elements: " + stateDescription.length);
        return -1;
      }
      
      List<List<Integer>> stateList = new ArrayList<>(lab.stateSpace);
      int stateIndex = stateList.indexOf(discretizedState);
      
      if (stateIndex == -1) {
        LOGGER.warning("State not found in state space: " + discretizedState);
        return 0; // Return first state as fallback
      }
      
      return stateIndex;
    } catch (Exception e) {
      LOGGER.severe("Error converting state description: " + e.getMessage());
      return -1;
    }
  }

  /**
   * Set default action when lookup fails
   */
  private void setDefaultAction(OpFeedbackParam<String> nextBestActionTag, 
                              OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                              OpFeedbackParam<Object[]> nextBestActionPayload) {
    LOGGER.info("Using default action");
    nextBestActionTag.set("http://example.org/was#SetZ1Light");
    Object[] payloadTags = { "Z1Light" };
    nextBestActionPayloadTags.set(payloadTags);
    Object[] payload = { true };
    nextBestActionPayload.set(payload);
  }

  /**
   * Export Q-table to CSV format
   */
  private void exportQTableToCSV(double[][] qTable, String filename, int goalHash) {
    try (FileWriter writer = new FileWriter(filename)) {
      writer.write("Goal Hash: " + goalHash + "\n");
      writer.write("States: " + qTable.length + ", Actions: " + qTable[0].length + "\n");
      writer.write("State\\Action");
      for (int a = 0; a < qTable[0].length; a++) {
        writer.write(",Action_" + a);
      }
      writer.write("\n");
      
      for (int s = 0; s < qTable.length; s++) {
        writer.write("State_" + s);
        for (int a = 0; a < qTable[s].length; a++) {
          writer.write("," + qTable[s][a]);
        }
        writer.write("\n");
      }
      
      LOGGER.info("Q-table exported to CSV: " + filename);
    } catch (IOException e) {
      LOGGER.severe("Failed to export Q-table to CSV: " + e.getMessage());
    }
  }

  /**
   * Export Q-table to text format
   */
  private void exportQTableToText(double[][] qTable, String filename, int goalHash) {
    try (FileWriter writer = new FileWriter(filename)) {
      writer.write("Q-TABLE EXPORT\n");
      writer.write("==============\n\n");
      writer.write("Goal Hash: " + goalHash + "\n");
      writer.write("Generated: " + new Date() + "\n");
      writer.write("State Space Size: " + qTable.length + "\n");
      writer.write("Action Space Size: " + qTable[0].length + "\n\n");
      
      writer.write(String.format("%-12s", "State\\Action"));
      for (int a = 0; a < qTable[0].length; a++) {
        writer.write(String.format("%12s", "A" + a));
      }
      writer.write("\n");
      writer.write("-".repeat(12 + qTable[0].length * 12) + "\n");
      
      // Show first 50 states to keep file manageable
      int maxStates = Math.min(50, qTable.length);
      for (int s = 0; s < maxStates; s++) {
        writer.write(String.format("%-12s", "S" + s));
        for (int a = 0; a < qTable[s].length; a++) {
          writer.write(String.format("%12.6f", qTable[s][a]));
        }
        writer.write("\n");
      }
      
      if (qTable.length > maxStates) {
        writer.write("\n... (showing first " + maxStates + " states of " + qTable.length + " total)\n");
      }
      
      LOGGER.info("Q-table exported to text: " + filename);
    } catch (IOException e) {
      LOGGER.severe("Failed to export Q-table to text: " + e.getMessage());
    }
  }

  /**
   * Export summary of all Q-tables
   */
  private void exportQTableSummary() {
    String filename = "qtable_summary_" + new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date()) + ".txt";
    
    try (FileWriter writer = new FileWriter(filename)) {
      writer.write("Q-TABLE SUMMARY REPORT\n");
      writer.write("======================\n\n");
      writer.write("Generated: " + new Date() + "\n");
      writer.write("Total Q-tables: " + qTables.size() + "\n\n");
      
      for (Map.Entry<Integer, double[][]> entry : qTables.entrySet()) {
        int goalHash = entry.getKey();
        double[][] qTable = entry.getValue();
        
        writer.write("Goal Hash: " + goalHash + "\n");
        writer.write("Dimensions: " + qTable.length + " × " + qTable[0].length + "\n");
        
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;
        double sum = 0.0;
        int nonZeroCount = 0;
        
        for (int i = 0; i < qTable.length; i++) {
          for (int j = 0; j < qTable[i].length; j++) {
            double value = qTable[i][j];
            max = Math.max(max, value);
            min = Math.min(min, value);
            sum += value;
            if (Math.abs(value) > 0.001) nonZeroCount++;
          }
        }
        
        writer.write("Max Q-value: " + String.format("%.6f", max) + "\n");
        writer.write("Min Q-value: " + String.format("%.6f", min) + "\n");
        writer.write("Avg Q-value: " + String.format("%.6f", sum / (qTable.length * qTable[0].length)) + "\n");
        writer.write("Learning coverage: " + String.format("%.1f%%", 100.0 * nonZeroCount / (qTable.length * qTable[0].length)) + "\n\n");
      }
      
      LOGGER.info("Q-table summary exported: " + filename);
    } catch (IOException e) {
      LOGGER.severe("Failed to export Q-table summary: " + e.getMessage());
    }
  }

  /**
   * Log Q-table statistics
   */
  private void logQTableStatistics(double[][] qTable, Object[] goalDescription) {
    double maxQ = Double.NEGATIVE_INFINITY;
    double minQ = Double.POSITIVE_INFINITY;
    double sumQ = 0.0;
    int nonZeroCount = 0;
    
    for (int i = 0; i < qTable.length; i++) {
      for (int j = 0; j < qTable[i].length; j++) {
        double qValue = qTable[i][j];
        maxQ = Math.max(maxQ, qValue);
        minQ = Math.min(minQ, qValue);
        sumQ += qValue;
        if (Math.abs(qValue) > 0.001) nonZeroCount++;
      }
    }
    
    double avgQ = sumQ / (qTable.length * qTable[0].length);
    
    LOGGER.info("Q-table statistics for goal " + Arrays.toString(goalDescription) + ":");
    LOGGER.info("  Max Q-value: " + String.format("%.2f", maxQ));
    LOGGER.info("  Min Q-value: " + String.format("%.2f", minQ));
    LOGGER.info("  Average Q-value: " + String.format("%.2f", avgQ));
    LOGGER.info("  Non-zero entries: " + nonZeroCount + "/" + (qTable.length * qTable[0].length));
  }

    /**
    * Print the Q matrix
    *
    * @param qTable the Q matrix
    */
  void printQTable(double[][] qTable) {
    System.out.println("Q matrix");
    for (int i = 0; i < Math.min(10, qTable.length); i++) { // Show first 10 states
      System.out.print("From state " + i + ":  ");
     for (int j = 0; j < qTable[i].length; j++) {
      System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
    if (qTable.length > 10) {
      System.out.println("... (showing first 10 states of " + qTable.length + " total)");
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