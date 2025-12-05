# Tech-Pulse Documentation Images

This directory contains screenshots and visual assets for the Tech-Pulse documentation.

## Image Descriptions

The following images are referenced in the documentation but need to be captured:

### 1. dashboard-overview.png
- **Location**: Referenced in README_USER.md
- **Description**: Main dashboard view showing metrics, charts, and data table
- **Size**: 1200x800px (recommended)
- **Format**: PNG with transparency support

### 2. semantic-search-interface.png
- **Location**: Referenced in user-guide.md
- **Description**: Semantic search tab with search input and results
- **Size**: 1200x800px (recommended)
- **Format**: PNG

### 3. sentiment-timeline.png
- **Description**: Sentiment analysis timeline chart showing mood trends
- **Size**: 800x400px (recommended)
- **Format**: PNG

### 4. topic-distribution.png
- **Description**: Pie chart showing topic distribution among stories
- **Size**: 600x600px (recommended)
- **Format**: PNG

### 5. control-panel-sidebar.png
- **Description**: Expanded sidebar showing all control options
- **Size**: 400x800px (recommended)
- **Format**: PNG

### 6. real-time-mode-indicator.png
- **Description**: Green indicator showing real-time mode is active
- **Size**: 300x100px (recommended)
- **Format**: PNG

### 7. pdf-briefing-preview.png
- **Description**: Preview of generated executive briefing PDF
- **Size**: 800x600px (recommended)
- **Format**: PNG

### 8. multi-source-settings.png
- **Description**: Multi-source mode settings with source selection
- **Size**: 400x300px (recommended)
- **Format**: PNG

### 9. predictive-analytics-tab.png
- **Description**: Predictive analytics dashboard with trend forecasts
- **Size**: 1200x800px (recommended)
- **Format**: PNG

### 10. mobile-responsive-view.png
- **Description**: Dashboard viewed on mobile device
- **Size**: 400x800px (recommended)
- **Format**: PNG

## Capturing Screenshots

### Tools Recommended
- **macOS**: Use built-in screenshot tool (Cmd+Shift+4)
- **Windows**: Use Snipping Tool or Snip & Sketch
- **Linux**: Use GNOME Screenshot or Flameshot

### Best Practices
1. **Clean Browser**: Remove bookmarks and extensions from view
2. **Consistent Theme**: Use light mode for all screenshots
3. **High Resolution**: Capture at 2x resolution when possible
4. **No Personal Data**: Ensure no sensitive information is visible
5. **Consistent Branding**: Use the same color scheme and styling

### Naming Convention
- Use lowercase with hyphens
- Be descriptive but concise
- Include feature name (e.g., semantic-search-results.png)

### Optimize for Web
- Use PNG for diagrams and interfaces
- Use JPEG for photographs
- Compress images to under 200KB when possible
- Use tools like TinyPNG for compression

## Image Usage in Documentation

### Markdown Syntax
```markdown
![Alt text](path/to/image.png)
```

### With Size Specifications
```markdown
![Dashboard Overview](images/dashboard-overview.png){ width=600 }
```

### With Captions
```markdown
<figure>
  <img src="images/dashboard-overview.png" alt="Tech-Pulse Dashboard">
  <figcaption>Fig 1: Main dashboard interface showing real-time tech news analysis</figcaption>
</figure>
```

## Accessibility

### Alt Text Requirements
- Describe the image content clearly
- Include relevant details for understanding
- Keep under 125 characters when possible

### Examples
**Good**: "Tech-Pulse dashboard showing sentiment analysis chart with green positive, gray neutral, and red negative segments"
**Bad**: "Dashboard image"

## Updating Images

When features change:
1. Capture new screenshots
2. Maintain consistent naming
3. Update alt text if needed
4. Check all documentation references
5. Commit with descriptive message

## Current Status

**Required**: 10 screenshots for complete documentation
**Status**: ⏳ Screenshots need to be captured

To capture screenshots:
1. Launch the dashboard: `streamlit run app.py`
2. Navigate through each feature
3. Capture high-quality screenshots
4. Optimize and save to this directory
5. Update references in documentation

---

*Note: Screenshots should reflect the current state of the application. Update them whenever UI changes are made.*