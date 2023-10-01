const express = require('express');
const dotenv = require('dotenv');
const router = require('./route');

dotenv.config();

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || 'localhost';

const app = express();

app.listen(PORT, () => {
  console.log(`App Started on ${PORT}`);
});

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const globalErrorHandler = (err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  const msg = err.message || 'Oops! something went wrong. Please try again';
  res.status(statusCode).send(msg);
};

app.use('/api/v1', router);

app.use(globalErrorHandler);

module.exports = app;
