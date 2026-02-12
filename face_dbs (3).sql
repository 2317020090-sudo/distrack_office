-- phpMyAdmin SQL Dump
-- version 5.2.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Feb 12, 2026 at 02:26 PM
-- Server version: 8.4.3
-- PHP Version: 8.3.16

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `face_dbs`
--

-- --------------------------------------------------------

--
-- Table structure for table `away_logs`
--

CREATE TABLE `away_logs` (
  `id` int NOT NULL,
  `name` varchar(100) NOT NULL,
  `exit_time` datetime NOT NULL,
  `return_time` datetime DEFAULT NULL,
  `duration_str` varchar(50) DEFAULT NULL,
  `total_minutes` int DEFAULT NULL,
  `evidence_path` varchar(255) DEFAULT NULL,
  `status_validasi` enum('PENDING','VALID','INVALID') DEFAULT 'PENDING'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `away_logs`
--

INSERT INTO `away_logs` (`id`, `name`, `exit_time`, `return_time`, `duration_str`, `total_minutes`, `evidence_path`, `status_validasi`) VALUES
(1, 'Paldo', '2026-02-09 14:34:23', '2026-02-09 14:34:43', '0:00:20', 0, 'captures/keluar_batas/AWAY_Paldo_1770622484.jpg', 'PENDING');

-- --------------------------------------------------------

--
-- Table structure for table `break_logs`
--

CREATE TABLE `break_logs` (
  `name` varchar(100) NOT NULL,
  `day_name` varchar(20) NOT NULL,
  `total_seconds` float DEFAULT '0',
  `exit_count` int DEFAULT '0',
  `last_updated` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `total_sleep_seconds` float DEFAULT '0',
  `total_likely_sleep_seconds` float DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `break_logs`
--

INSERT INTO `break_logs` (`name`, `day_name`, `total_seconds`, `exit_count`, `last_updated`, `total_sleep_seconds`, `total_likely_sleep_seconds`) VALUES
('Paldo', 'Kamis', 12.883, 2, '2026-02-12 07:55:42', 0, 0),
('Paldo', 'Selasa', 0, 0, '2026-02-10 14:39:24', 0, 0),
('Paldo', 'Senin', 0, 0, '2026-02-09 09:01:05', 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `daily_attendance`
--

CREATE TABLE `daily_attendance` (
  `id` int NOT NULL,
  `name` varchar(100) NOT NULL,
  `date` date NOT NULL,
  `time_in` time DEFAULT NULL,
  `time_out` time DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `daily_attendance`
--

INSERT INTO `daily_attendance` (`id`, `name`, `date`, `time_in`, `time_out`) VALUES
(1, 'Paldo', '2026-02-12', '13:32:15', NULL);

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int NOT NULL,
  `name` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `unit` varchar(100) DEFAULT NULL,
  `role` varchar(100) DEFAULT NULL,
  `password_hash` varchar(255) NOT NULL,
  `embedding` text NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `violation_sessions`
--

CREATE TABLE `violation_sessions` (
  `id` int NOT NULL,
  `name` varchar(100) NOT NULL,
  `violation_type` enum('TIDUR','KEMUNGKINAN TIDUR') NOT NULL,
  `start_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `end_time` timestamp NULL DEFAULT NULL,
  `start_image_path` varchar(255) DEFAULT NULL,
  `end_image_path` varchar(255) DEFAULT NULL,
  `duration_str` varchar(50) DEFAULT NULL,
  `admin_status` enum('PENDING','VALID','INVALID') DEFAULT 'PENDING',
  `status_sesi` enum('ONGOING','FINISHED') DEFAULT 'ONGOING'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `violation_sessions`
--

INSERT INTO `violation_sessions` (`id`, `name`, `violation_type`, `start_time`, `end_time`, `start_image_path`, `end_image_path`, `duration_str`, `admin_status`, `status_sesi`) VALUES
(4, 'Paldo', 'TIDUR', '2026-02-10 04:12:16', '2026-02-10 04:12:18', 'captures/tidur/START_Paldo_1770696736.jpg', 'captures/active_after_sleep/END_Paldo_1770696738.jpg', '0:00:12', 'PENDING', 'FINISHED');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `away_logs`
--
ALTER TABLE `away_logs`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `break_logs`
--
ALTER TABLE `break_logs`
  ADD PRIMARY KEY (`name`,`day_name`);

--
-- Indexes for table `daily_attendance`
--
ALTER TABLE `daily_attendance`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `unique_attendance` (`name`,`date`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `violation_sessions`
--
ALTER TABLE `violation_sessions`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `away_logs`
--
ALTER TABLE `away_logs`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `daily_attendance`
--
ALTER TABLE `daily_attendance`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `violation_sessions`
--
ALTER TABLE `violation_sessions`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
