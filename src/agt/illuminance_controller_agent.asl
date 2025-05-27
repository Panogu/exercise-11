//illuminance controller agent - Hybrid Version (Learn in Sim, Act in Real)

/* Initial beliefs and rules */
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").  // Simulation for learning
real_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab-real.ttl").  // Real lab for actions
task_requirements([2,3]).

/* Initial goals */
!start.

/* 
 * Main startup plan - Hybrid Implementation
 */
+!start : learning_lab_environment(SimUrl) 
  & real_lab_environment(RealUrl)
  & task_requirements([Z1Level, Z2Level]) <-

  .print("=== Hybrid Illuminance Controller Agent Starting ===");
  .print("Learning on: ", SimUrl);
  .print("Acting on: ", RealUrl);
  .print("Target: Z1Level=", Z1Level, " Z2Level=", Z2Level);

  /* Create the QLearner artifact using SIMULATION environment */
  makeArtifact("qlearner", "tools.QLearner", [SimUrl], QLArtId);
  
  /* Create the ThingArtifact using REAL environment for actions */
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [RealUrl], LabArtId);
  
  .print("=== Starting Q-Learning Training (Simulation) ===");
  
  /* Train Q-Learning with simulation environment */
  calculateQ([Z1Level, Z2Level], 100, 0.2, 0.9, 0.3, 1000);
  .print("Primary Q-Learning completed for goal [", Z1Level, ",", Z2Level, "]");
  
  /* Train for additional goals for comparison */
  calculateQ([3, 3], 150, 0.2, 0.9, 0.3, 1000);
  .print("Q-Learning completed for goal [3,3]");
  
  calculateQ([1, 1], 150, 0.2, 0.9, 0.3, 1000);
  .print("Q-Learning completed for goal [1,1]");
  
  .print("=== Q-Learning Training Complete ===");
  
  /* Export Q-tables for submission */
  exportAllQTables;
  .print("Q-tables exported");
  
  .print("=== Now Acting on REAL Environment ===");
  
  /* Start acting on the REAL environment using learned policy */
  !achieve_goal_real.

/*
 * Goal achievement plan for REAL environment
 */
+!achieve_goal_real : task_requirements([Z1Level, Z2Level]) <-
  .print("=== REAL ENVIRONMENT: Attempting to reach goal [", Z1Level, ",", Z2Level, "] ===");
  !act_towards_goal_real(0).

/*
 * Acting plan for REAL environment - uses REAL lab ThingArtifact but LEARNED policy
 */
+!act_towards_goal_real(Step) : Step < 10 <-  // Fewer steps for real environment
  .print("--- REAL ACTION Step ", Step, " ---");
  
  /* Read current state from REAL environment using ThingArtifact */
  readProperty("https://example.org/was#Status", CurrentRealState);
  .print("REAL lab current state: ", CurrentRealState);
  
  /* Convert real state to discretized format for Q-learning */
  !convert_real_state_to_discrete(CurrentRealState, DiscreteState);
  
  /* Get best action using learned Q-table from simulation */
  task_requirements([Z1Level, Z2Level]);
  getActionFromState([Z1Level, Z2Level], DiscreteState, ActionTag, PayloadTags, Payload);
  .print("REAL environment - Selected action: ", ActionTag, " with payload: ", Payload);
  
  /* Execute the action on REAL environment */
  invokeAction(ActionTag, PayloadTags, Payload);
  .print("REAL action executed successfully");
  
  /* Wait for real environment to update */
  .wait(5000);  // Longer wait for real environment
  
  /* Continue to next step */
  NextStep = Step + 1;
  !act_towards_goal_real(NextStep).

/*
 * Plan when maximum steps reached in real environment
 */
+!act_towards_goal_real(Step) : Step >= 10 <-
  .print("=== REAL ENVIRONMENT: Reached maximum steps (10) ===");
  readProperty("https://example.org/was#Status", FinalRealState);
  .print("Final achieved state in REAL lab: ", FinalRealState);
  .print("=== REAL Environment Goal Achievement Complete ===").

/*
 * Helper plan to convert real lab state to discrete format expected by Q-learner
 */
+!convert_real_state_to_discrete(RealState, DiscreteState) <-
  // Extract values from real state JSON and discretize them
  // This might need adjustment based on the actual format you receive
  // For now, assuming similar structure to simulation
  DiscreteState = RealState.  // Simplified - you might need more processing

// ... keep all other existing plans for learning, exporting, etc.