import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Grid,
  Chip,
} from '@mui/material';
import {
  Download,
  TableChart,
  Image,
  Refresh,
} from '@mui/icons-material';
import axios from 'axios';

const Results = () => {
  const [csvData, setCsvData] = useState(null);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [availableFiles, setAvailableFiles] = useState([]);

  useEffect(() => {
    // In a real implementation, you'd fetch available result files
    // For now, we'll show a placeholder
  }, []);

  const loadCsvData = async (filename) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`/api/csv-data/${filename}`);
      setCsvData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to load CSV data');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const downloadFile = (filename) => {
    window.open(`/api/download/${filename}`, '_blank');
  };

  const renderTable = () => {
    if (!csvData) return null;

    return (
      <TableContainer component={Paper} sx={{ mt: 2, backgroundColor: 'rgba(26, 31, 58, 0.8)' }}>
        <Table>
          <TableHead>
            <TableRow>
              {csvData.columns.map((column) => (
                <TableCell key={column} sx={{ fontWeight: 'bold' }}>
                  {column}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {csvData.data.slice(0, 100).map((row, index) => (
              <TableRow key={index}>
                {csvData.columns.map((column) => (
                  <TableCell key={column}>
                    {typeof row[column] === 'number' 
                      ? row[column].toFixed(4) 
                      : row[column] || '-'
                    }
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Script Results
        </Typography>
        <Typography variant="body1" color="text.secondary">
          View and download results from your script executions
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab icon={<TableChart />} label="CSV Results" />
          <Tab icon={<Image />} label="Charts & Images" />
        </Tabs>
      </Box>

      {tabValue === 0 && (
        <Box>
          <Card sx={{ mb: 3, backgroundColor: 'rgba(26, 31, 58, 0.8)' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CSV Data Files
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Load and view CSV results from your script executions
              </Typography>
              
              {/* Sample file buttons - in real implementation, these would be dynamic */}
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => loadCsvData('sample_results.csv')}
                  disabled={loading}
                >
                  Load Sample Results
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Refresh />}
                  onClick={() => window.location.reload()}
                >
                  Refresh
                </Button>
              </Box>

              {loading && (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              )}

              {csvData && (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1">
                      Showing {Math.min(100, csvData.total_rows)} of {csvData.total_rows} rows
                    </Typography>
                    <Button
                      variant="contained"
                      startIcon={<Download />}
                      onClick={() => downloadFile('sample_results.csv')}
                      sx={{ backgroundColor: '#00d4aa' }}
                    >
                      Download CSV
                    </Button>
                  </Box>
                  {renderTable()}
                </Box>
              )}

              {!csvData && !loading && (
                <Alert severity="info">
                  No CSV data loaded. Run a script to generate results, then click "Load Sample Results" to view them here.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Box>
      )}

      {tabValue === 1 && (
        <Box>
          <Card sx={{ backgroundColor: 'rgba(26, 31, 58, 0.8)' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Charts & Images
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                View generated charts and analysis images
              </Typography>

              <Grid container spacing={2}>
                {/* Sample image placeholders */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ backgroundColor: 'rgba(45, 53, 97, 0.5)' }}>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        Sample Chart 1
                      </Typography>
                      <Box
                        sx={{
                          height: 200,
                          backgroundColor: 'rgba(0, 0, 0, 0.2)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: 1,
                          mb: 2,
                        }}
                      >
                        <Typography color="text.secondary">
                          Chart will appear here after running scripts
                        </Typography>
                      </Box>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Download />}
                        disabled
                      >
                        Download Image
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Card sx={{ backgroundColor: 'rgba(45, 53, 97, 0.5)' }}>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        Sample Chart 2
                      </Typography>
                      <Box
                        sx={{
                          height: 200,
                          backgroundColor: 'rgba(0, 0, 0, 0.2)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: 1,
                          mb: 2,
                        }}
                      >
                        <Typography color="text.secondary">
                          Chart will appear here after running scripts
                        </Typography>
                      </Box>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Download />}
                        disabled
                      >
                        Download Image
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              <Alert severity="info" sx={{ mt: 3 }}>
                Charts and images will appear here after running scripts that generate visual outputs.
              </Alert>
            </CardContent>
          </Card>
        </Box>
      )}
    </Container>
  );
};

export default Results;