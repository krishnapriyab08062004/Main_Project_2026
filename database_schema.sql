-- Database Schema for Speech Emotion Recognition System

-- 1. Create the database (if not exists)
CREATE DATABASE IF NOT EXISTS ser;
USE ser;

-- 2. Create the 'users' table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create the 'emotion_logs' table
CREATE TABLE IF NOT EXISTS emotion_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    emotion VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    stress_score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
