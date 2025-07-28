import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Box,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Chip,
  Divider,
} from '@mui/material';
import {
  PlayArrow,
  ArrowBack,
  Settings,
  CheckCircle,
  Error,
  Schedule,
} from '@mui/icons-material';
import axios from 'axios';

const ScriptRunner = () => {
  const { category } = useParams();
  const navigate = useNavigate();
  const [scripts, setScripts] = useState({});
  const [selectedScript, setSelectedScript] = useState(null);
  const [params, setParams] = useState({});
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [runDialog, setRunDialog] = useState(false);

  useEffect(() => {
    fetchScripts();
  }, []);

  useEffect(() => {
    let interval;
    if (jobId && jobStatus?.status === 'running') {
      interval = setInterval(checkJobStatus, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, jobStatus?.status]);

  const fetchScripts = async () => {
    try {
      const response = await axios.get('/api/scripts');
      setScripts(response.data[category] || {});
      setLoading(false);
    } catch (err) {
      setError('Failed to load scripts');
      setLoading(false);
    }
  };

  const checkJobStatus = async () => {
    if (!jobId) return;
    
    try {
      const response = await axios.get(`/api/job-status/${jobId}`);
      setJobStatus(response.data);
      
      if (response.data.status !== 'running') {
        setRunning(false);
      }
    } catch (err) {
      console.error('Error checking job status:', err);
    }
  };

  const handleParamChange = (paramName, value) => {
    setParams(prev => ({
      ...prev,
      [paramName]: value
    }));
  };

  const runScript = async (scriptName) => {
    setRunning(true);
    setError(null);
    setJobStatus(null);
    
    try {
      const response = await axios.post('/api/run-script', {
        category,
        script: scriptName,
        params
      });
      
      setJobId(response.data.job_id);
      setRunDialog(true);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to run script');
      setRunning(false);
    }
  };

  const formatCategoryName = (category) => {
    return category
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' & ');
  };

  const renderParamInput = (param) => {
    switch (param.type) {
      case 'select':
        return (
          <FormControl fullWidth margin="normal">
            <InputLabel>{param.name}</InputLabel>
            <Select
              value={params[param.name] || param.default || ''}
              label={param.name}
              onChange={(e) => handleParamChange(param.name, e.target.value)}
            >
              {param.options.map(option => (
                <MenuItem key={option} value={option}>{option}</MenuItem>
              ))}
            </Select>
          </FormControl>
        );
      case 'number':
        return (
          <TextField
            fullWidth
            margin="normal"
            label={param.name}
            type="number"
            value={params[param.name] || param.default || ''}
            onChange={(e) => handleParamChange(param.name, e.target.value)}
            inputProps={{
              min: param.min,
              max: param.max
            }}
          />
        );
      case 'textarea':
        return (
          <TextField
            fullWidth
            margin="normal"
            label={param.name}
            multiline
            rows={3}
            value={params[param.name] || param.default || ''}
            onChange={(e) => handleParamChange(param.name, e.target.value)}
            placeholder="Enter comma-separated values"
          />
        );
      default:
        return (
          <TextField
            fullWidth
            margin="normal"
            label={param.name}
            value={params[param.name] || param.default || ''}
            onChange={(e) => handleParamChange(param.name, e.target.value)}
            required={param.required}
          />
        );
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'info';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'timeout': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <Schedule />;
      case 'completed': return <CheckCircle />;
      case 'failed': return <Error />;
      case 'timeout': return <Error />;
      default: return null;
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          Back to Dashboard
        </Button>
        <Typography variant="h4" component="h1" gutterBottom>
          {formatCategoryName(category)} Scripts
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Select and configure a script to run
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {Object.entries(scripts).map(([scriptName, scriptInfo]) => (
          <Grid item xs={12} md={6} key={scriptName}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                background: 'linear-gradient(135deg, rgba(26, 31, 58, 0.8) 0%, rgba(45, 53, 97, 0.8) 100%)',
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="h6" component="h2" gutterBottom>
                  {scriptInfo.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {scriptInfo.description}
                </Typography>

                {scriptInfo.params && scriptInfo.params.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                      <Settings sx={{ mr: 1, fontSize: 16 }} />
                      Parameters
                    </Typography>
                    {scriptInfo.params.map((param) => (
                      <Box key={param.name} sx={{ mb: 1 }}>
                        {renderParamInput(param)}
                      </Box>
                    ))}
                  </Box>
                )}
              </CardContent>
              
              <CardActions sx={{ p: 2, pt: 0 }}>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={() => runScript(scriptName)}
                  disabled={running}
                  startIcon={running ? <CircularProgress size={16} /> : <PlayArrow />}
                  sx={{
                    backgroundColor: '#00d4aa',
                    '&:hover': {
                      backgroundColor: '#00b894',
                    },
                  }}
                >
                  {running ? 'Running...' : 'Run Script'}
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Job Status Dialog */}
      <Dialog open={runDialog} onClose={() => setRunDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Script Execution Status
          {jobStatus && (
            <Chip
              label={jobStatus.status}
              color={getStatusColor(jobStatus.status)}
              icon={getStatusIcon(jobStatus.status)}
              sx={{ ml: 2 }}
            />
          )}
        </DialogTitle>
        <DialogContent>
          {jobStatus?.status === 'running' && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Script is running, please wait...
              </Typography>
              <LinearProgress />
            </Box>
          )}

          {jobStatus?.output && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Output:</Typography>
              <Box
                sx={{
                  backgroundColor: 'rgba(0, 0, 0, 0.3)',
                  p: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  maxHeight: 200,
                  overflow: 'auto',
                }}
              >
                <pre>{jobStatus.output}</pre>
              </Box>
            </Box>
          )}

          {jobStatus?.error && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom color="error">
                Error:
              </Typography>
              <Box
                sx={{
                  backgroundColor: 'rgba(255, 0, 0, 0.1)',
                  p: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  maxHeight: 200,
                  overflow: 'auto',
                }}
              >
                <pre>{jobStatus.error}</pre>
              </Box>
            </Box>
          )}

          {jobStatus?.results && jobStatus.results.length > 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Results:
              </Typography>
              {jobStatus.results.map((result, index) => (
                <Box key={index} sx={{ mb: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => window.open(`/api/download/${result.filename}`, '_blank')}
                  >
                    Download {result.filename}
                  </Button>
                </Box>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRunDialog(false)}>Close</Button>
          {jobStatus?.status === 'completed' && (
            <Button
              variant="contained"
              onClick={() => {
                setRunDialog(false);
                navigate('/results');
              }}
            >
              View Results
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ScriptRunner;