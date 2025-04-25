import React, { useState } from 'react';
import { Upload, Card, Form, Input, Select, Button, Spin, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Dragger } = Upload;
const { Option } = Select;

function App() {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [form] = Form.useForm();

  const handleSubmit = async (values) => {
    if (!values.file || !values.file[0]) {
      message.error('Please upload an X-ray image');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', values.file[0].originFileObj);
    formData.append('metadata', JSON.stringify({
      patient_id: values.patient_id,
      view_position: values.view_position,
      patient_gender: values.patient_gender,
      patient_age: values.patient_age
    }));

    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setReport(response.data);
    } catch (error) {
      message.error('Error generating report: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <h1>XrayGPT Radiograph Analysis</h1>
      
      <Form form={form} onFinish={handleSubmit} layout="vertical">
        <Form.Item name="file" label="X-ray Image" required>
          <Dragger maxCount={1}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">Click or drag X-ray image to upload</p>
          </Dragger>
        </Form.Item>

        <Form.Item name="patient_id" label="Patient ID" rules={[{ required: true }]}>
          <Input />
        </Form.Item>

        <Form.Item name="view_position" label="View Position" rules={[{ required: true }]}>
          <Select>
            <Option value="PA">PA</Option>
            <Option value="AP">AP</Option>
            <Option value="LATERAL">LATERAL</Option>
          </Select>
        </Form.Item>

        <Form.Item name="patient_gender" label="Gender" rules={[{ required: true }]}>
          <Select>
            <Option value="M">Male</Option>
            <Option value="F">Female</Option>
          </Select>
        </Form.Item>

        <Form.Item name="patient_age" label="Age" rules={[{ required: true }]}>
          <Input type="number" />
        </Form.Item>

        <Form.Item>
          <Button type="primary" htmlType="submit" disabled={loading}>
            Generate Report
          </Button>
        </Form.Item>
      </Form>

      {loading && <Spin tip="Generating report..." />}

      {report && (
        <Card title="Generated Report" style={{ marginTop: 24 }}>
          <p><strong>Confidence:</strong> {(report.confidence * 100).toFixed(1)}%</p>
          <p><strong>Report:</strong></p>
          <pre>{report.report}</pre>
          <p><strong>Findings:</strong></p>
          <ul>
            {report.findings.map((finding, index) => (
              <li key={index}>{finding}</li>
            ))}
          </ul>
          <p><strong>Impression:</strong> {report.impression}</p>
        </Card>
      )}
    </div>
  );
}

export default App;