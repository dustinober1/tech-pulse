# Tech-Pulse Streamlit Cloud Deployment Guide

## Prerequisites Checklist

Before deploying to Streamlit Cloud, ensure the following items are complete:

- [ ] All tests passing (120 tests, 92.5% pass rate)
- [ ] requirements.txt is complete (11 packages)
- [ ] .gitignore is configured (comprehensive patterns)
- [ ] .streamlit/config.toml exists (valid TOML)
- [ ] Git repository pushed to GitHub (https://github.com/dustinober1/tech-pulse)

## Deployment Steps

Follow these steps to deploy Tech-Pulse to Streamlit Cloud:

1. **Navigate to Streamlit Cloud**
   - Go to https://share.streamlit.io

2. **Sign In**
   - Sign in with your GitHub account
   - Authorize Streamlit to access your repositories

3. **Create New App**
   - Click the "New app" button

4. **Configure Repository Settings**
   - **Repository**: Select `dustinober1/tech-pulse`
   - **Branch**: `main`
   - **Main file path**: `app.py`

5. **Deploy**
   - Click "Deploy" button
   - Wait for deployment process to complete

## Expected Behavior During Deployment

During the deployment process, you should observe the following:

- **Dependency Installation**: Initial install of dependencies from requirements.txt
- **NLTK Data Download**: Automatic download of vader_lexicon
- **Cache Directory Creation**: System creates cache/ directory for data persistence
- **Configuration Loading**: Streamlit loads settings from .streamlit/config.toml
- **App Launch**: App should be live at https://tech-pulse.streamlit.app (or similar URL)

## Success Indicators

Once deployment is complete, verify the following:

- [ ] App loads without errors
- [ ] Dashboard displays Hacker News stories
- [ ] Sentiment analysis results are visible
- [ ] Topic modeling visualization is present
- [ ] No "Failed to initialize" messages appear
- [ ] Cache system is functioning (subsequent loads are faster)

## Troubleshooting Common Issues

### Memory Issues

**Problem**: BERTopic can be memory-intensive on free tier

**Solution**:
- Consider using lighter models if needed
- Reduce the number of stories processed
- Optimize model parameters in `models/topic_model.py`

### NLTK Data Download Failures

**Problem**: NLTK vader_lexicon fails to download

**Solution**:
- Already handled in `data_loader.py` with automatic retry logic
- Usually resolves automatically on retry
- Check logs for specific error messages

### Cache Directory Errors

**Problem**: Cache directory cannot be created

**Solution**:
- `cache/` directory is created automatically by the app
- Excluded from git via .gitignore
- Streamlit Cloud provides writable `/tmp` directory if needed

### Python Version Compatibility

**Problem**: Dependency conflicts or version issues

**Details**:
- Requires Python 3.9-3.11
- Streamlit Cloud uses Python 3.11 by default
- All dependencies in requirements.txt are compatible

**Solution**:
- Verify requirements.txt has correct versions
- Add `python-version` file if specific version needed

### Port/Binding Issues

**Problem**: App fails to bind to correct port

**Solution**:
- Already configured for headless mode in `.streamlit/config.toml`
- Port 8501 set for standard Streamlit deployment
- Streamlit Cloud handles port binding automatically

### Import Errors

**Problem**: Missing dependencies or module not found

**Solution**:
- Verify all dependencies are listed in requirements.txt
- Check that package names are correct (e.g., `scikit-learn` not `sklearn`)
- Review deployment logs for specific missing packages

## Support and Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Streamlit Community Forum**: https://discuss.streamlit.io
- **Repository**: https://github.com/dustinober1/tech-pulse
- **Verification Script**: `python scripts/verify_deployment.py <URL>`

## Post-Deployment Verification

After deployment is complete, run the verification script to ensure everything is working correctly:

```bash
python scripts/verify_deployment.py https://tech-pulse.streamlit.app
```

This script validates:
- App is accessible
- API endpoints respond correctly
- Data loading works
- Sentiment analysis functions
- Topic modeling displays properly

## Monitoring and Maintenance

### App Logs

Access logs through the Streamlit Cloud dashboard:
1. Go to https://share.streamlit.io
2. Select your app
3. Click "Logs" to view real-time application logs

### App Analytics

Monitor usage and performance:
1. View analytics in Streamlit Cloud dashboard
2. Track visitor counts and session duration
3. Monitor resource usage (RAM, CPU)

### Updating the Deployment

To update your deployed app:
1. Push changes to the `main` branch on GitHub
2. Streamlit Cloud automatically detects changes
3. App redeploys automatically (auto-deploy enabled by default)

Or manually trigger redeployment:
1. Go to Streamlit Cloud dashboard
2. Select your app
3. Click "Reboot" to force redeploy

## Security Considerations

- **API Keys**: Use Streamlit Secrets for any API keys (not currently needed)
- **Data Privacy**: HN data is public, no sensitive information stored
- **HTTPS**: Streamlit Cloud provides HTTPS by default
- **Dependencies**: Keep requirements.txt updated with security patches

## Performance Optimization

For optimal performance on Streamlit Cloud:

1. **Caching**: Already implemented with `@st.cache_data` decorators
2. **Data Loading**: Stories cached for 1 hour to reduce API calls
3. **Model Caching**: BERTopic models cached to avoid retraining
4. **Session State**: Used efficiently to maintain state across reruns

## Rollback Procedure

If deployment fails or issues arise:

1. **Revert Git Commit**:
   ```bash
   git revert HEAD
   git push origin main
   ```

2. **Or Deploy Specific Commit**:
   - In Streamlit Cloud dashboard
   - Click "Advanced settings"
   - Specify commit SHA to deploy

## Additional Configuration

### Custom Domain (Optional)

To use a custom domain:
1. Upgrade to Streamlit Cloud paid tier
2. Follow custom domain setup instructions
3. Configure DNS settings as directed

### Resource Limits

Free tier limits:
- **RAM**: 1 GB
- **CPU**: Shared
- **Storage**: Limited to /tmp directory
- **Apps**: 3 public apps

## Conclusion

Your Tech-Pulse app is now deployed and accessible to users worldwide. Monitor the app regularly and use the verification script to ensure continued functionality. For issues or questions, refer to the support resources listed above.
