1. User Interface Improvement:
  - Add functions:
    - Create New Chat
    - Find and enter previous chat
  - More modern design, smooth UI
  - In the beginning, there would only be one search panel for user at the center
  - Keep the suggested question bubbles below the center at the beginning of the chat
  - Stream the agent response to improve user experience
  - Keep user updated about what the agent is doing right now, showing status such as "thinking...", "searching...", "finding the best matches...", etc.
  - The return from the agent is usually in Markdown format, you should make sure the UI supports markdown.
  - Instead of using bubbles to show conversations, use something more modernized, without bubble, but with clear separation between user and agent.
2. Guardrail:
  - on failure:
    - tool return specific error back to agent (if timeout it should retry 1 time)
    - agent retry 1 time on failure
    - if still not working, agent tells user something goes wrong
    - write into logfile (record with date time)
  - Before app backend starts totally, the front end will wait.
3. Performance:
  - image and text embedding pipelines should be asynchronized, with two threads, to speed up (check if it really can speed up or not)
  - reranker should be ran on batching / multithreading to speed up (we should decide which implementation is more efficient, or both)
4. Test:
  - Create a test sample list in csv, using 100 samples for each test scenarios (image-based, text-based, follow-ups), it should include test type, test query, test id (specifying multi round tests), expected results.
  - Create a test file for testing models on the test sample list, and tell me how to use it in test md.
