⚡ Phase 5: The Live Wire (Real-Time Updates)

**Goal**: Transform your static dashboard into a self-updating "Command Center."
**Output**: An updated app.py that refreshes data automatically every 60 seconds without a full page reload.
**Prerequisites**: You must have Phase 3 working (basic dashboard).

📝 Context

Currently, your app only updates when you click "Refresh." In a real Ops/Trading/News dashboard, data should flow alive.

We will use st.empty() containers. These are special slots in Streamlit that can be overwritten by new data while the rest of the page stays still. We will wrap our main logic in a while True loop that sleeps for 60 seconds between fetches.

---

## 🗺️ Detailed Implementation Roadmap

### **Work Package 1: Infrastructure Preparation** (Foundation)

#### **Task 5.1.1: Update Session State for Real-Time Mode**
- **Files**: `app.py` (lines 23-36)
- **Implementation**:
  - Add `real_time_mode` boolean to `initialize_session_state()`
  - Remove existing `auto_refresh` and `refresh_countdown` variables
  - Add `last_update_time` to track timestamp
- **Success Criteria**: Session state includes new real-time variables

#### **Task 5.1.2: Update Sidebar with Real-Time Toggle**
- **Files**: `app.py` (lines 75-82)
- **Implementation**:
  - Replace auto-refresh checkbox with "Enable Real-Time Mode" toggle
  - Update help text to reflect 60-second refresh interval
  - Add visual indicator when real-time mode is active
- **Success Criteria**: Sidebar shows real-time toggle with appropriate labeling

### **Work Package 2: Core Real-Time Logic** (Engine)

#### **Task 5.2.1: Create Real-Time Display Container**
- **Files**: `app.py` (new function)
- **Implementation**:
  - Create `create_realtime_display()` function
  - Initialize `placeholder = st.empty()` at appropriate scope
  - Structure content to render within placeholder context
- **Success Criteria**: Function creates placeholder container for content

#### **Task 5.2.2: Implement While True Loop Logic**
- **Files**: `app.py` (main function)
- **Implementation**:
  - Check if real-time mode is enabled
  - If enabled: enter `while True` loop
  - Inside loop: use `with placeholder.container():`
  - Fetch data, analyze, and display metrics
  - Add `time.sleep(60)` at loop end
  - Handle exceptions gracefully without breaking loop
- **Success Criteria**: Real-time mode updates every 60 seconds automatically

#### **Task 5.2.3: Add Timestamp Display**
- **Files**: `app.py` (create_header or new function)
- **Implementation**:
  - Import datetime if not already imported
  - Create `display_timestamp()` function
  - Show timestamp in top-right corner with caption
  - Format as HH:MM:SS for easy readability
  - Display in both real-time and manual modes
- **Success Criteria**: Timestamp updates every refresh cycle

### **Work Package 3: Dual Mode Operation** (UX)

#### **Task 5.3.1: Preserve Manual Refresh Mode**
- **Files**: `app.py` (main function flow)
- **Implementation**:
  - Create conditional logic: if real-time mode disabled
  - Use existing static display functions
  - Keep refresh button functionality intact
  - Ensure timestamp updates on manual refresh
- **Success Criteria**: Manual refresh works exactly as before when real-time is off

#### **Task 5.3.2: Implement Smooth Mode Transition**
- **Files**: `app.py`
- **Implementation**:
  - Add state management for mode switching
  - Clear placeholder when switching modes
  - Prevent UI freezing during transitions
  - Maintain data consistency across modes
- **Success Criteria**: Switching modes doesn't require page reload

### **Work Package 4: Testing & Quality Assurance** (Quality)

#### **Task 5.4.1: Write Real-Time Logic Tests**
- **Files**: `test/test_realtime_features.py` (new)
- **Implementation**:
  - Test session state initialization for real-time variables
  - Test real-time toggle functionality
  - Mock time.sleep for loop testing
  - Test placeholder container creation
  - Test timestamp display accuracy
- **Success Criteria**: All real-time features have unit test coverage

#### **Task 5.4.2: Integration Tests for Real-Time Dashboard**
- **Files**: `test/test_realtime_integration.py` (new)
- **Implementation**:
  - Test full real-time refresh cycle
  - Test mode switching scenarios
  - Test error handling in real-time loop
  - Test UI responsiveness during updates
  - Test concurrent operations
- **Success Criteria**: Integration tests pass for all real-time scenarios

#### **Task 5.4.3: Performance Tests**
- **Files**: `test/test_realtime_performance.py` (new)
- **Implementation**:
  - Test memory usage during extended real-time sessions
  - Test API rate limiting compliance
  - Test browser tab behavior with long-running sessions
  - Test UI responsiveness during updates
- **Success Criteria**: No memory leaks or performance degradation

### **Work Package 5: Documentation & Deployment** (Finalization)

#### **Task 5.5.1: Update README with Real-Time Features**
- **Files**: `README.md`
- **Implementation**:
  - Add Real-Time Features section
  - Document how to enable/disable real-time mode
  - Add troubleshooting guide for real-time issues
  - Update screenshots if needed
- **Success Criteria**: README clearly explains real-time functionality

#### **Task 5.5.2: Update Dashboard Configuration**
- **Files**: `dashboard_config.py`
- **Implementation**:
  - Add REAL_TIME_SETTINGS dictionary
  - Add real-time help text
  - Update error messages for real-time failures
  - Add success messages for real-time activation
- **Success Criteria**: Configuration supports real-time settings

#### **Task 5.5.3: Commit and Push Changes**
- **Implementation**:
  - Stage all modified and new files
  - Create descriptive commit message
  - Push changes to remote repository
- **Success Criteria**: All changes committed and pushed

---

## 📅 Execution Timeline

```
Week 1:
- Day 1: Tasks 5.1.1, 5.1.2 (Parallel)
- Day 2: Task 5.2.1
- Day 3: Task 5.2.2
- Day 4: Task 5.2.3
- Day 5: Task 5.3.1

Week 2:
- Day 1: Task 5.3.2
- Day 2: Tasks 5.4.1, 5.4.2 (Parallel)
- Day 3: Task 5.4.3
- Day 4: Tasks 5.5.1, 5.5.2 (Parallel)
- Day 5: Task 5.5.3
```

---

## ⚡ Original AI Prompts (Legacy Reference)

### Task 5.1: The "Auto-Refresh" Logic

**Agent Prompt**:
"Refactor my app.py to support real-time updates.

Session State: Add a toggle in the sidebar: real_time = st.sidebar.checkbox('Enable Real-Time Mode', value=False).

The Loop:
If real_time is Checked:
- Create a placeholder using placeholder = st.empty()
- Enter a while True loop
- Inside the loop, context manage the placeholder: with placeholder.container():
- Call fetch_hn_data, analyze_sentiment, and get_topics
- Render all my metrics and charts inside this container
- Add time.sleep(60) at the end of the loop

If real_time is Unchecked:
- Keep the original 'Refresh Button' logic I had before

Important: Ensure the app doesn't freeze the UI entirely (Streamlit handles sleep okay, but warn me if I need st_autorefresh instead)."

### Task 5.2: The "Last Updated" Timestamp

**Agent Prompt**:
"Add a small timestamp indicator.
- Import datetime
- Inside the display logic (both real-time and manual), add a small caption at the top right: st.caption(f'Last Updated: {datetime.now().strftime("%H:%M:%S")}')
- This verifies that the loop is actually working."

---

## ✅ Success Criteria

Run `streamlit run app.py`.

Check the "Enable Real-Time Mode" box in the sidebar.

**Win Conditions**:
- The "Last Updated" time changes every minute automatically
- You do not have to click anything to refresh data
- If you leave the tab open and come back 10 minutes later, the data (and charts) have changed
- Manual refresh mode still works when real-time is disabled
- All tests pass with >90% code coverage

---

## ⚠️ Important Considerations

### **API Rate Limiting**
- **Hacker News API**: Very generous, but polling every 60 seconds is fine
- **Do not go lower than 10 seconds** or you might get blocked

### **Streamlit Cloud Behavior**
- Apps often "go to sleep" if no one is looking at them to save resources
- Real-time loops might eventually timeout on the free tier
- This is normal behavior for free hosting

### **Performance Considerations**
- Monitor memory usage during extended real-time sessions
- Implement proper error handling to prevent loop breaks
- Ensure UI remains responsive during updates

---

## 🚀 Key Success Metrics

- Real-time updates occur every 60 seconds without user interaction
- Manual refresh mode remains fully functional
- Timestamp accurately reflects last update
- No UI freezing or performance degradation
- All tests pass with comprehensive coverage
- Smooth switching between real-time and manual modes