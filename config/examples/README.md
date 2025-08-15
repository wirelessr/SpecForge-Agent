# Configuration Examples

This directory contains example configuration files for different use cases.

## Framework Configuration Examples

### comprehensive-example.json
Complete example showing all available configuration options with explanations:
- All configuration sections with default values
- Comprehensive comments explaining each setting
- Good starting point for custom configurations

### development.json
Configuration optimized for development work:
- Higher verbosity for debugging
- Moderate timeouts and retries
- Smaller memory limits for faster iteration
- Pattern learning enabled for improvement

### production.json
Configuration optimized for production deployment:
- Lower temperature for consistent results
- Higher token limits for complex tasks
- Longer timeouts for reliability
- Larger memory limits for better context
- Verbose mode disabled for cleaner logs

### testing.json
Configuration optimized for automated testing:
- Very low temperature for deterministic results
- Auto-approve enabled for unattended execution
- Short timeouts for fast test execution
- Minimal memory usage
- Session reset on start for clean state

## Model Configuration Examples

### custom-models.json
Example of adding custom model configurations:
- Custom provider models with specific token limits
- Local model configurations
- Pattern matching for model families
- Different capability sets per model type

## Usage

To use these example configurations:

1. **Copy to main config directory**:
   ```bash
   cp config/examples/development.json config/framework.json
   ```

2. **Override via environment variable**:
   ```bash
   export FRAMEWORK_CONFIG_FILE=config/examples/production.json
   ```

3. **Override via command line**:
   ```bash
   autogen-framework --framework-config config/examples/testing.json
   ```

## Customization

Feel free to modify these examples or create your own based on your specific needs:

1. **Copy an example** that's closest to your use case
2. **Modify settings** as needed
3. **Test thoroughly** with your specific models and workflows
4. **Document changes** for your team

## Validation

Before using custom configurations, validate the JSON syntax:

```bash
python -m json.tool config/examples/your-config.json
```